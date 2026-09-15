"""Offline tests for the distributed shared cache (app/shared_cache.py).

All tests run against mock / local backends — no network required.
"""
import json
import threading
import time
import unittest
from unittest import mock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from app.shared_cache import (
    CacheStats,
    LocalLRUCache,
    TieredCache,
    UpstashRedisCache,
    get_cache,
    reset_cache_for_tests,
)
import app.config as config


class TestCacheStats(unittest.TestCase):
    """Verify hit-rate arithmetic."""

    def test_empty_stats(self) -> None:
        s = CacheStats()
        self.assertEqual(s.total, 0)
        self.assertAlmostEqual(s.hit_rate, 0.0)
        snap = s.snapshot()
        self.assertEqual(snap["total"], 0)
        self.assertAlmostEqual(snap["hit_rate"], 0.0)

    def test_hit_rate_calculation(self) -> None:
        s = CacheStats(hits=7, misses=3, errors=1, sets=5, evictions=2)
        self.assertEqual(s.total, 10)
        self.assertAlmostEqual(s.hit_rate, 0.7)
        snap = s.snapshot()
        self.assertEqual(snap["hits"], 7)
        self.assertEqual(snap["errors"], 1)
        self.assertEqual(snap["sets"], 5)
        self.assertEqual(snap["evictions"], 2)


class TestLocalLRUCache(unittest.TestCase):
    """Test the process-local LRU fallback."""

    def setUp(self) -> None:
        self.cache = LocalLRUCache(maxsize=4)

    def test_basic_set_get(self) -> None:
        self.cache.set("k1", "v1")
        self.assertEqual(self.cache.get("k1"), "v1")

    def test_miss_returns_none(self) -> None:
        self.assertIsNone(self.cache.get("nonexistent"))

    def test_hit_miss_tracking(self) -> None:
        self.cache.set("k1", "v1")
        self.cache.get("k1")  # hit
        self.cache.get("miss")  # miss
        self.assertEqual(self.cache.stats.hits, 1)
        self.assertEqual(self.cache.stats.misses, 1)

    def test_eviction_when_full(self) -> None:
        for i in range(5):
            self.cache.set(f"k{i}", f"v{i}")
        # k0 should have been evicted (maxsize=4).
        self.assertIsNone(self.cache.get("k0"))
        self.assertEqual(self.cache.stats.evictions, 1)
        # Most recent should still be there.
        self.assertEqual(self.cache.get("k4"), "v4")

    def test_lru_ordering(self) -> None:
        for i in range(4):
            self.cache.set(f"k{i}", f"v{i}")
        # Access k0 to make it recently used.
        self.cache.get("k0")
        # Adding k4 should evict k1 (oldest unused).
        self.cache.set("k4", "v4")
        self.assertIsNone(self.cache.get("k1"))
        self.assertEqual(self.cache.get("k0"), "v0")

    def test_delete(self) -> None:
        self.cache.set("k1", "v1")
        self.cache.delete("k1")
        self.assertIsNone(self.cache.get("k1"))

    def test_exists(self) -> None:
        self.cache.set("k1", "v1")
        self.assertTrue(self.cache.exists("k1"))
        self.assertFalse(self.cache.exists("miss"))

    def test_ttl_expiry(self) -> None:
        self.cache.set("k1", "v1", ttl_seconds=0)  # expires immediately
        time.sleep(0.01)
        self.assertIsNone(self.cache.get("k1"))

    def test_is_shared_false(self) -> None:
        self.assertFalse(self.cache.is_shared)

    def test_health_report(self) -> None:
        self.cache.set("k1", "v1")
        self.cache.get("k1")
        h = self.cache.health()
        self.assertEqual(h["backend"], "local_lru")
        self.assertFalse(h["is_shared"])
        self.assertEqual(h["size"], 1)
        self.assertEqual(h["maxsize"], 4)
        self.assertEqual(h["hits"], 1)

    def test_thread_safety(self) -> None:
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(50):
                    self.cache.set(f"t{i}", f"v{i}")
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                for i in range(50):
                    self.cache.get(f"t{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        self.assertEqual(errors, [])


class TestUpstashRedisCache(unittest.TestCase):
    """Test the Upstash REST client with mocked HTTP responses."""

    def setUp(self) -> None:
        self.cache = UpstashRedisCache(
            url="https://test.upstash.io",
            token="test-token",
            timeout_seconds=1.0,
        )

    def test_get_returns_value(self) -> None:
        resp = {"result": "hello"}
        with mock.patch("app.shared_cache.urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = mock.Mock(return_value=mock.Mock(read=mock.Mock(return_value=json.dumps(resp).encode())))
            mock_open.return_value.__exit__ = mock.Mock(return_value=False)
            result = self.cache.get("mykey")
        self.assertEqual(result, "hello")
        self.assertEqual(self.cache.stats.hits, 1)

    def test_get_returns_none_on_missing(self) -> None:
        resp = {"result": None}
        with mock.patch("app.shared_cache.urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = mock.Mock(return_value=mock.Mock(read=mock.Mock(return_value=json.dumps(resp).encode())))
            mock_open.return_value.__exit__ = mock.Mock(return_value=False)
            result = self.cache.get("miss")
        self.assertIsNone(result)
        self.assertEqual(self.cache.stats.misses, 1)

    def test_get_returns_none_on_network_error(self) -> None:
        with mock.patch("app.shared_cache.urllib.request.urlopen", side_effect=OSError("timeout")):
            result = self.cache.get("key")
        self.assertIsNone(result)
        self.assertEqual(self.cache.stats.errors, 1)

    def test_set_calls_post(self) -> None:
        resp = {"result": "OK"}
        with mock.patch("app.shared_cache.urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = mock.Mock(return_value=mock.Mock(read=mock.Mock(return_value=json.dumps(resp).encode())))
            mock_open.return_value.__exit__ = mock.Mock(return_value=False)
            self.cache.set("k", "v", ttl_seconds=60)
        self.assertEqual(self.cache.stats.sets, 1)

    def test_is_shared_true(self) -> None:
        self.assertTrue(self.cache.is_shared)

    def test_health_report(self) -> None:
        # ping will fail (no real server) — that's expected.
        h = self.cache.health()
        self.assertEqual(h["backend"], "upstash_redis")
        self.assertTrue(h["is_shared"])
        self.assertFalse(h["reachable"])


class TestTieredCache(unittest.TestCase):
    """Test the two-tier cache with mock shared + local."""

    def setUp(self) -> None:
        reset_cache_for_tests()
        self.local = LocalLRUCache(maxsize=10)
        self.shared = mock.MagicMock(spec=UpstashRedisCache)
        self.shared.is_shared = True
        self.shared.stats = CacheStats()
        self.tiered = TieredCache(shared=self.shared, local=self.local)

    def tearDown(self) -> None:
        reset_cache_for_tests()

    def test_read_shared_hit(self) -> None:
        self.shared.get.return_value = "from_redis"
        result = self.tiered.get("k1")
        self.assertEqual(result, "from_redis")
        # Should back-fill local.
        self.assertEqual(self.local.get("k1"), "from_redis")

    def test_read_shared_miss_falls_to_local(self) -> None:
        self.shared.get.return_value = None
        self.local.set("k1", "from_local")
        result = self.tiered.get("k1")
        self.assertEqual(result, "from_local")

    def test_read_shared_error_falls_to_local(self) -> None:
        self.shared.get.side_effect = OSError("timeout")
        self.local.set("k1", "fallback")
        result = self.tiered.get("k1")
        self.assertEqual(result, "fallback")
        self.assertEqual(self.tiered.shared_stats.errors, 1)

    def test_write_populates_both_tiers(self) -> None:
        self.tiered.set("k1", "v1", ttl_seconds=30)
        self.shared.set.assert_called_once_with("k1", "v1", ttl_seconds=30)
        self.assertEqual(self.local.get("k1"), "v1")

    def test_delete_removes_from_both(self) -> None:
        self.local.set("k1", "v1")
        self.tiered.delete("k1")
        self.shared.delete.assert_called_once_with("k1")
        self.assertIsNone(self.local.get("k1"))

    def test_exists_checks_shared_first(self) -> None:
        self.shared.exists.return_value = True
        self.assertTrue(self.tiered.exists("k1"))

    def test_exists_falls_to_local(self) -> None:
        self.shared.exists.return_value = False
        self.local.set("k1", "v1")
        self.assertTrue(self.tiered.exists("k1"))

    def test_combined_stats(self) -> None:
        self.shared.get.side_effect = lambda k: "v" if k == "k1" else None
        self.tiered.get("k1")  # shared hit
        self.tiered.get("k2")  # shared miss → local miss
        # Combined hits: shared_stats.hits (1) + local_stats.hits (0) = 1
        # Combined misses: local_stats.misses (1) only (shared miss falls through)
        self.assertEqual(self.tiered.stats.hits, 1)
        self.assertEqual(self.tiered.stats.misses, 1)

    def test_is_shared_true_when_shared_present(self) -> None:
        self.assertTrue(self.tiered.is_shared)

    def test_is_shared_false_when_no_shared(self) -> None:
        tiered_only = TieredCache(shared=None, local=self.local)
        self.assertFalse(tiered_only.is_shared)

    def test_health_report(self) -> None:
        self.shared.health.return_value = {"backend": "upstash_redis", "is_shared": True}
        h = self.tiered.health()
        self.assertTrue(h["tiered"])
        self.assertIn("shared", h)
        self.assertIn("local", h)

    def test_health_report_local_only(self) -> None:
        tiered_only = TieredCache(shared=None, local=self.local)
        h = tiered_only.health()
        self.assertFalse(h["tiered"])
        self.assertNotIn("shared", h)


class TestSingletonFactory(unittest.TestCase):
    """Test get_cache() singleton and reset."""

    def test_get_cache_returns_same_instance(self) -> None:
        reset_cache_for_tests()
        c1 = get_cache()
        c2 = get_cache()
        self.assertIs(c1, c2)

    def test_reset_creates_new_instance(self) -> None:
        c1 = get_cache()
        reset_cache_for_tests()
        c2 = get_cache()
        self.assertIsNot(c1, c2)

    def test_singleton_with_env_vars(self) -> None:
        reset_cache_for_tests()
        with mock.patch.dict(config.__dict__, {
            "SHARED_CACHE_URL": "https://test.upstash.io",
            "SHARED_CACHE_TOKEN": "test-token",
            "SHARED_CACHE_TIMEOUT_SECONDS": 5.0,
        }):
            cache = get_cache()
            self.assertTrue(cache.is_shared)
            self.assertTrue(cache.health()["tiered"])

    def test_singleton_without_env_vars(self) -> None:
        reset_cache_for_tests()
        with mock.patch.dict(config.__dict__, {
            "SHARED_CACHE_URL": "",
            "SHARED_CACHE_TOKEN": "",
        }):
            cache = get_cache()
            self.assertFalse(cache.is_shared)


class TestCacheIntegration(unittest.TestCase):
    """End-to-end: set → get → hit rate → health across the tiered cache."""

    def test_full_lifecycle(self) -> None:
        reset_cache_for_tests()
        with mock.patch.dict(config.__dict__, {
            "SHARED_CACHE_URL": "",
            "SHARED_CACHE_TOKEN": "",
        }):
            cache = get_cache()

            # Set several keys.
            for i in range(5):
                cache.set(f"key{i}", f"val{i}", ttl_seconds=60)

            # Read them back.
            for i in range(5):
                self.assertEqual(cache.get(f"key{i}"), f"val{i}")

            # Check hit rate.
            stats = cache.stats
            self.assertEqual(stats.hits, 5)
            self.assertEqual(stats.sets, 5)
            self.assertGreater(stats.hit_rate, 0.0)

            # Health report.
            h = cache.health()
            self.assertIn("combined", h)
            self.assertIn("local", h)

            reset_cache_for_tests()


if __name__ == "__main__":
    unittest.main()
