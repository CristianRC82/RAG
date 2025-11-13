import os, json, tempfile
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorstoreManager:


    def __init__(self, gcs, embeddings):
        self.gcs = gcs
        self.embeddings = embeddings
        self.folder_prefix = os.getenv("GCS_FOLDER_PATH")
        self.vectorstore_path_remote = os.getenv("GCS_VECTORSTORE_PATH")
        self.chroma_persist_directory = tempfile.mkdtemp()
        self.index_file_path = os.path.join(self.chroma_persist_directory, "index_state.json")

    def _load_previous_index(self):
        if os.path.exists(self.index_file_path):
            with open(self.index_file_path, "r") as f:
                return json.load(f)
        return {}

    def _build_current_index(self):
        index = {}
        for blob in self.gcs.client.list_blobs(self.gcs.bucket_name, prefix=self.folder_prefix):
            if blob.name.endswith((".pdf", ".txt")):
                index[blob.name] = {"updated": str(blob.updated), "size": blob.size}
        return index

    def _detect_changes(self, prev, current):
        added = [f for f in current if f not in prev]
        removed = [f for f in prev if f not in current]
        modified = [f for f in current if f in prev and (
            current[f]["updated"] != prev[f]["updated"] or current[f]["size"] != prev[f]["size"]
        )]
        return any([added, removed, modified])

    def load_or_create_vectorstore(self):
        existing_files = list(self.gcs.client.list_blobs(self.gcs.bucket_name, prefix=self.vectorstore_path_remote))
        vectorstore_exists = bool(existing_files)

        if vectorstore_exists:
            self.gcs.download_folder_to_local(self.vectorstore_path_remote, self.chroma_persist_directory)

        prev_index = self._load_previous_index()
        current_index = self._build_current_index()
        has_changes = self._detect_changes(prev_index, current_index)

        if vectorstore_exists and not has_changes:
            return Chroma(persist_directory=self.chroma_persist_directory, embedding_function=self.embeddings)

        return self._rebuild_vectorstore(current_index)

    def _rebuild_vectorstore(self, current_index):
        downloaded_files = self.gcs.download_folder(self.folder_prefix)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["<br>"]
            )
        documents = []

        for file_path in downloaded_files:
            if file_path.endswith(".pdf"):
                docs = PyPDFLoader(file_path).load_and_split()
            elif file_path.endswith(".txt"):
                text_docs = TextLoader(file_path, encoding="utf-8").load()
                docs = splitter.split_documents(text_docs)
            else:
                continue
            documents.extend(docs)

        vectorstore = Chroma.from_documents(documents, self.embeddings, persist_directory=self.chroma_persist_directory)
        with open(self.index_file_path, "w") as f:
            json.dump(current_index, f, indent=2)

        self.gcs.upload_folder(self.chroma_persist_directory, self.vectorstore_path_remote)
        return vectorstore
