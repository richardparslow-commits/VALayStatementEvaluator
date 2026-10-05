"""Synthetic evidence-integrity regressions for source block boundaries."""
from __future__ import annotations

import hashlib
import random
import unittest
from unittest.mock import patch

from tests import hermetic  # noqa: F401
from tests.ingestion_fixtures import docx_parts, package
from tests.test_parser_boundary import IMAGE, request
from app import config, documents, pilot
from app.documents import (ChunkPlan, DocumentPage, ExtractedDocument, ExtractionError,
                           _blocks_from_text, chunk_page_labelled_text, document_from_text,
                           extract_document)
from app.isolated_extract import _documents
from app.job_payload import (PayloadError, document_from_json, document_to_json,
                            request_documents_from_json)
from app.medical_review import _dates_in_text
from app.parser_protocol import ParserRefused
from app.views import uploads


def reply(doc, req):
    return {**{k: v for k, v in req.items() if k not in ('size', 'page_limit')},
            'image': IMAGE, 'documents': [document_to_json(doc)], 'skipped': []}


class SourceSpanTests(unittest.TestCase):
    def assert_source(self, doc, source, limit):
        self.assertEqual(doc.full_text, source)
        self.assertEqual(''.join(p.text for p in doc.pages), source)
        cursor = 0
        for page in doc.pages:
            self.assertEqual(page.source_start, cursor)
            self.assertEqual(page.source_end, cursor + len(page.text))
            self.assertEqual(page.text, source[page.source_start:page.source_end])
            self.assertLessEqual(len(page.text), limit)
            cursor = page.source_end
        self.assertEqual(cursor, len(source))
        self.assertEqual(doc.char_count, len(source))

    def test_random_whitespace_and_paragraph_layouts_reconstruct_exactly(self):
        rng = random.Random(507)
        for case in range(120):
            limit = rng.randint(16, 128)
            tokens = ['Synthetic', '2020-01-13', 'denies', '-12.75mg', 'cafe\u0301']
            source = ' \t' + ''.join(rng.choice(tokens) + rng.choice([' ', '\t', '\r\n', '\n\n\n'])
                                      for _ in range(rng.randint(1, 90))) + '  '
            with self.subTest(case=case), patch.object(config, 'DOCUMENT_BLOCK_CHARS', limit):
                self.assert_source(document_from_text('record.txt', source), source, limit)

    def test_date_dose_negation_and_unicode_tokens_survive_every_seam_offset(self):
        limit = 96
        tokens = ('2020-01-13', '01/13/2020', '-12.75', '\u22120.5mg', 'denies',
                  'hypothyroidism', 'cafe\u0301', '\U0001f469\u200d\u2695\ufe0f')
        for token in tokens:
            for offset in range(1, len(token)):
                with self.subTest(token=token, offset=offset), patch.object(config, 'DOCUMENT_BLOCK_CHARS', limit):
                    prefix_len = limit - offset
                    prefix = ('pad ' * (prefix_len // 4)) + (' ' * (prefix_len % 4))
                    source = prefix + token + ' confirmed in the synthetic source.'
                    doc = document_from_text('record.txt', source)
                    self.assert_source(doc, source, limit)
                    self.assertTrue(any(token in p.text for p in doc.pages))
                    self.assertIn(token, doc.page_labelled_text())
                    self.assertTrue(any(token in c.text for c in ChunkPlan(doc.pages)))
                    self.assertEqual(_dates_in_text(doc.full_text), _dates_in_text(source))

    def test_whitespace_is_preserved_separately_from_normalized_prompt_views(self):
        source = '  \tSynthetic note\r\n\r\nNo\t\tdiagnosis.\n\n\nDose: -0.5 mg.  \t'
        with patch.object(config, 'DOCUMENT_BLOCK_CHARS', 24):
            doc = extract_document('record.md', source.encode())
        self.assert_source(doc, source, 24)
        self.assertEqual(doc.source_sha256, hashlib.sha256(source.encode()).hexdigest())
        self.assertNotEqual(doc.page_labelled_text(), source)

    def test_raw_source_spans_have_identical_streamed_and_joined_model_chunks(self):
        source = '  \tSynthetic denial.\r\n\r\n' + ('Dose -0.5 mg; no diagnosis.  ' * 190) + '\t '
        with patch.object(config, 'DOCUMENT_BLOCK_CHARS', 96):
            doc = document_from_text('record.txt', source)
        for budget in (1200, 3000):
            joined = chunk_page_labelled_text(doc.page_labelled_text(), max_chars=budget)
            streamed = list(ChunkPlan(doc.pages, max_chars=budget))
            self.assertEqual([(c.text, c.pages) for c in streamed], [(c.text, c.pages) for c in joined])
        self.assertEqual(doc.full_text, source)

    def test_blank_source_spans_survive_parser_and_saved_request_roundtrips(self):
        source = (' \t' * 40) + 'Synthetic final denial.'
        with patch.object(config, 'DOCUMENT_BLOCK_CHARS', 24):
            doc = extract_document('record.txt', source.encode())
        self.assertTrue(any(not p.text.strip() for p in doc.pages))
        req = {**request(), 'size': len(source.encode()), 'sha256': doc.source_sha256}
        parsed, skipped = _documents(reply(doc, req), req, IMAGE)
        self.assertEqual(skipped, [])
        self.assert_source(parsed[0], source, 24)
        restored = request_documents_from_json([document_to_json(parsed[0])])[0]
        self.assert_source(restored, source, 24)

    def test_docx_stories_keep_independent_offsets_and_distinct_source_parts(self):
        main = 'Synthetic history. ' * 10 + '2020-01-13: no diagnosis.'
        header = 'Synthetic header: denies symptoms.'
        parts = docx_parts(main)
        parts['word/header1.xml'] = ('<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                                     '<w:p><w:r><w:t>' + header + '</w:t></w:r></w:p></w:hdr>').encode()
        with patch.object(config, 'DOCUMENT_BLOCK_CHARS', 48):
            doc = extract_document('record.docx', package(parts))
        for part, source in (('word/document.xml', main), ('word/header1.xml', header)):
            pages = [p for p in doc.pages if p.source_part == part]
            self.assert_source(ExtractedDocument(doc.filename, pages), source, 48)
        self.assertEqual(doc.full_text, main + '\n\n' + header)
        req = {**request(), 'label': doc.filename, 'sha256': doc.source_sha256}
        parsed = _documents(reply(doc, req), req, IMAGE)[0][0]
        restored = document_from_json(document_to_json(parsed))
        self.assertEqual(restored.full_text, doc.full_text)
        self.assertEqual([(p.source_part, p.source_start, p.source_end) for p in restored.pages],
                         [(p.source_part, p.source_start, p.source_end) for p in doc.pages])

    def test_over_limit_source_tokens_refuse_whole_documents_and_archive_members(self):
        with patch.object(config, 'DOCUMENT_BLOCK_CHARS', 24):
            for source in ('X' * 25, 'Synthetic note. ' + 'X' * 25):
                with self.subTest(source_len=len(source)), self.assertRaisesRegex(ExtractionError, 'source token'):
                    document_from_text('record.txt', source)
            docs, skipped = documents.InProcessExtractor().extract('records.zip', package({
                'valid.txt': b'Synthetic denial.', 'oversized.txt': b'X' * 25}))
        self.assertEqual([d.filename for d in docs], ['records/valid.txt'])
        self.assertEqual(len(skipped), 1)
        self.assertIn('oversized.txt', skipped[0])
        self.assertIn('source token', skipped[0])

    def test_non_positive_or_non_integer_block_limits_are_refused(self):
        for limit in (0, -1, True, 1.5, '24'):
            with self.subTest(limit=limit), self.assertRaises(ExtractionError):
                _blocks_from_text('record.txt', 'Synthetic text.', limit)

    def test_physical_pdf_page_boundaries_still_have_separators(self):
        doc = ExtractedDocument('record.pdf', [DocumentPage('record.pdf', 1, 'First page.'),
                                             DocumentPage('record.pdf', 2, 'Second page.')])
        self.assertEqual(doc.full_text, 'First page.\n\nSecond page.')

    def test_corrupt_offsets_are_refused_by_parser_and_saved_job_decoders(self):
        with patch.object(config, 'DOCUMENT_BLOCK_CHARS', 24):
            doc = extract_document('record.txt', b'Synthetic initial denial. Synthetic final denial.')
        req = {**request(), 'sha256': doc.source_sha256}
        for index, field, value in ((0, 'source_start', 1), (0, 'source_start', True),
                                    (0, 'source_end', None), (0, 'source_end', 999999),
                                    (1, 'source_start', doc.pages[1].source_start + 1),
                                    (1, 'source_start', doc.pages[1].source_start - 1)):
            data = document_to_json(doc)
            data['pages'][index][field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ParserRefused):
                    _documents({**reply(doc, req), 'documents': [data]}, req, IMAGE)
                with self.assertRaises(PayloadError):
                    document_from_json(data)

    def test_contiguous_offsets_cannot_authorize_a_split_source_token(self):
        doc = extract_document('record.txt', b'2020-01-13')
        data = document_to_json(doc)
        data.update(total_pages=2, pagination='block')
        data['pages'] = [{'page': 1, 'kind': 'block', 'source_part': '', 'text': '2020-', 'source_start': 0, 'source_end': 5},
                         {'page': 2, 'kind': 'block', 'source_part': '', 'text': '01-13', 'source_start': 5, 'source_end': 10}]
        req = {**request(), 'sha256': doc.source_sha256}
        with self.assertRaises(ParserRefused):
            _documents({**reply(doc, req), 'documents': [data]}, req, IMAGE)
        with self.assertRaises(PayloadError):
            document_from_json(data)

    def test_saved_source_pages_refuse_coercion_and_reordered_addresses(self):
        with patch.object(config, 'DOCUMENT_BLOCK_CHARS', 24):
            doc = extract_document('record.txt', b'Synthetic initial denial. Synthetic final denial.')
        for field, bad in (('text', 123), ('page', True), ('page', 2), ('source_part', 123)):
            data = document_to_json(doc)
            data['pages'][0][field] = bad
            with self.subTest(field=field), self.assertRaises(PayloadError):
                document_from_json(data)

    def test_legacy_parser_schema_is_refused_and_saved_blocks_have_unknown_coverage(self):
        doc = extract_document('record.txt', b'Synthetic denied symptoms.')
        data = document_to_json(doc)
        data.update(schema_version=3, pagination='block')
        for page in data['pages']:
            page['kind'] = 'block'
            del page['source_start'], page['source_end']
        restored = document_from_json(data)
        self.assertFalse(restored.coverage_known)
        req = {**request(), 'sha256': doc.source_sha256}
        with self.assertRaises(ParserRefused):
            _documents({**reply(doc, req), 'documents': [data]}, req, IMAGE)

    def test_legacy_standalone_and_archive_caches_are_recomputed(self):
        from tests.test_archive_accounting import Uploaded
        for filename in ('record.txt', 'record.md', 'record.docx', 'records.zip'):
            with self.subTest(filename=filename), patch.object(config, 'DOCUMENT_BLOCK_CHARS', 48):
                source = 'Synthetic history. ' * 8 + '2020-01-13: no diagnosis.'
                body = package(docx_parts(source)) if filename.endswith('.docx') else source.encode()
                if filename.endswith('.zip'):
                    body = package({'record.txt': body})
                uploaded = Uploaded(body)
                uploaded.name = filename
                digest = hashlib.sha256(body).hexdigest()
                old_key = f'eval:{filename}:{len(body)}:{digest}'
                if filename.endswith('.zip'):
                    old_key = f'eval:all-members-v1:{filename}:{len(body)}:{digest}'
                state = {old_key: {'documents': [document_from_text('legacy.txt', 'Incomplete synthetic source.')], 'skipped': []}}
                with patch.object(documents, '_ACTIVE_EXTRACTOR', documents.InProcessExtractor()), \
                        patch.object(uploads.st, 'session_state', state), patch.object(pilot, 'enabled', return_value=False):
                    result = uploads.extract_uploads([uploaded], 'eval')
                self.assertEqual(result[0].full_text, source)
                self.assertNotIn(old_key, state)
                self.assertEqual(len(state), 1)


if __name__ == '__main__':
    unittest.main()
