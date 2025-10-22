# chat/management/commands/build_index.py
from django.core.management.base import BaseCommand
from backend.data_loader import load_all_documents
from backend.vectorstore import FaissVectorStore
from django.conf import settings

class Command(BaseCommand):
    help = "Build (or rebuild) the FAISS index from documents in ./data"

    def handle(self, *args, **options):
        # Step 1: Load all docs from your data folder
        self.stdout.write("[INFO] Loading documents...")
        docs = load_all_documents("data")
        self.stdout.write(f"[INFO] Loaded {len(docs)} documents.")

        # Step 2: Initialize FAISS store
        store = FaissVectorStore(settings.FAISS_DIR)

        # Step 3: Build the index
        self.stdout.write("[INFO] Building FAISS index...")
        store.build_from_documents(docs)

        # Step 4: Optionally save it
        if hasattr(store, "save"):
            store.save()
        elif hasattr(store, "persist"):
            store.persist()

        self.stdout.write(self.style.SUCCESS("✅ FAISS index built successfully."))
