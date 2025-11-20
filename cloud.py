import os
import tempfile
from dotenv import load_dotenv
from google.cloud import storage

class GoogleCloudStorageManager:

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.bucket_name = os.getenv("GCS_BUCKET_NAME")

        if not self.credentials_path or not os.path.exists(self.credentials_path):
            raise FileNotFoundError("No se encontró el archivo de credenciales JSON definido en .env")

        self.client = storage.Client.from_service_account_json(self.credentials_path)
        self.bucket = self.client.bucket(self.bucket_name)

    def download_folder(self, prefix: str) -> list:
        blobs = list(self.client.list_blobs(self.bucket_name, prefix=prefix))
        temp_dir = tempfile.mkdtemp()
        downloaded_files = []

        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            filename = os.path.basename(blob.name)
            local_path = os.path.join(temp_dir, filename)
            blob.download_to_filename(local_path)
            downloaded_files.append(local_path)
        return downloaded_files

    def download_folder_to_local(self, prefix: str, local_dir: str):
        os.makedirs(local_dir, exist_ok=True)
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)

        for blob in blobs:
            if blob.name.endswith("/"):
                continue

            relative_path = blob.name[len(prefix):]
            local_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            print(f"{blob.name} → {local_path}")

    def upload_folder(self, local_dir: str, remote_prefix: str):
        for root, _, files in os.walk(local_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_dir)
                blob_path = os.path.join(remote_prefix, relative_path).replace("\\", "/")
                self.bucket.blob(blob_path).upload_from_filename(local_path)
                print(f"Subido: {blob_path}")

    def list_files(self, prefix: str = ""):
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
        print(f"\nArchivos en gs://{self.bucket_name}/{prefix}:")
        for blob in blobs:
            print(f" - {blob.name}")
