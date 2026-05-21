"""
Run from the project root:
    python reindex_docs.py
"""

import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
sys.path.insert(0, os.path.dirname(__file__))
django.setup()

from documents.models import Document, DocumentEmbedding
from documents.signals import _ensure_worker, _index_queue, SUPPORTED_EXTENSIONS

DOCUMENT_IDS = [133, 134]

for doc_id in DOCUMENT_IDS:
    try:
        doc = Document.objects.get(id=doc_id)
    except Document.DoesNotExist:
        print(f'[SKIP] Document {doc_id} not found.')
        continue

    if not doc.file:
        print(f'[SKIP] Document {doc_id} ({doc.title!r}) has no file attached.')
        continue

    ext = os.path.splitext(doc.file.path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(f'[SKIP] Document {doc_id} ({doc.title!r}) — unsupported type: {ext}')
        continue

    # Reset embedding status so the worker picks it up
    DocumentEmbedding.objects.filter(document=doc).update(
        index_status='pending',
        is_indexed=False,
        error_message='',
    )

    _ensure_worker()
    _index_queue.put(doc.id)
    print(f'[QUEUED] Document {doc_id}: {doc.title!r} ({ext})')

# Wait for the queue to drain before exiting
print('\nWaiting for indexing to finish...')
_index_queue.join()
print('Done.')
