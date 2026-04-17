import os
from sentence_transformers import SentenceTransformer
import uuid
import chromadb
from typing import Dict, List, Optional
from chromadb.config import Settings
from azure.storage.blob import BlobServiceClient, ContentSettings


class VectorDBManager:
    """
    Hybrid storage manager:
    - Job description :stores JD chunking and embeddings
    - Candidate CV : Stores candidate cv embeddings and chunks
    - Raw files in the Azure blob storage
    - Embeddings and chunks in the local disk
    """
    def __init__(self,
                 persist_directory: str = "../chroma_db",
                 azure_container_name: str = "resume-screning",
                 azure_connection_string: str = None
    ):
        """
        Initialize the chromadb and vector db manager
        
        Args:
            persistent_directory : where to storage database
            azure_container_name: str = "resume-screning
            azure_connection_string str = None
        """

        # intialize the chroma db client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # intialize the vector embeddings
        print("loading embedding model and this will take a few mins")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-V2')
        print("embedding loaded")

        # create collections for cv and jd
        #now create the jd collection
        self.jd_collection=self.client.get_or_create_collection(
                name="job_descriptions",
                metadata={"description": "JD embeddings"}
        )
        
        self.cv_collections=self.client.get_or_create_collection(
                name = "Resumes",
                metadata={"description": "Candidate resume embeddings"}
        )
        # initilize the blob strorage
        if azure_connection_string:
            self.blob_service_client=BlobServiceClient.from_connection_string(
                    azure_connection_string
        )
            self.container_client=self.blob_service_client.get_container_client(
                    container_name
        )

            try:
                self.container_client.create_container()
                print(f"Created Azure container: {ontainer_name}")
            except Exception as e:
                print(f"Container {container_name} already exists or error: {e}")

            else:
                self.blob_service_client = none
                self.container_client=none
                print("Azure Blob Storage not configured (running in local mode)")

    def upload_to_blob(self, file_path: str, blob_name: str) -> str:
        """
        Upload original file to Azure Blob Storage
        Args:
            file_path: Path to local file
            blob_name: Name to use in blob storage

        Returns:
            Blob_url
        """

        if not self.container_client:
            print("Azure Blob Storage not configured, skipping upload")
            return file_path

        try:
            with open(file_path, "rb") as data:
                blob_client = self.container_client.get_blob_client(blob_name)
                blob_client.upload_blob(data, overwrite=True)

            blob_url = blob_client.url
            print(f"✓ Uploaded {blob_name} to Azure Blob Storage")
            return blob_url
             

        except Exception as e:
            print(f"Error uploading to blob: {e}")
            return file_path

    def download_from_blob(self, blob_name: str, download_path: str) -> str:
     
       """
        Download file from Azure Blob Storage
        Args:
           blob_name: Name of blob to download
           download_path: Where to save locally
        
       Returns:
           Local file path
       """

       if not self.container_client:
           Print("Azure Blob Storage not configured")
           return download_path


       try:
           blob_client = self.container_client.get_blob_client(blob_name)
           os.makedirs(os.path.dirname(download_path), exist_ok=True)

           with open(download_path, "wb") as download_file:
               download_stream = blob_client.download_blob()
               download_file.write(download_stream.readall())

               print(f"✓ Downloaded {blob_name} from Azure Blob Storage")
               return download_path

       except Exception as e:
           print(f"Azure Blob Storage not configured:{e}")

           return none

    def store_jd(
           self,
           chunks: List[Dict],
           jd_id: Optional[str] =None,
           role: Optional[str] = None,
           original_file_path: Optional[str]=None
    )-> str:
       """
       Store JD chunks in vector database and original file in Azure Blob

       Args:
           chunks: List of chunk dictionaries with 'content' and 'metadata'
           jd_id: Optional JD identifier (auto-generated if not provided)
           role: Job role/title
           original_file_path: Path to original PDF/DOCX file
       Returns:
           jd_id for reference
       """

       if not jd_id:
           jd_id = str(uuid.uuid4())

        # Upload original file to Azure Blob Storage

       blob_url=None
       if original_file_path and self.container_client:
           blob_name=f"jds/{jd_id}/{os.path.basename(original_file_path)}"
           blob_url=self.upload_to_blob(original_file_path, blob_name)

           # extract text from vhunks
           text= [chunks["content"] for chunk in chunks]

           # generage embeddings
           print(f"The embeddings for the text are:{len(texts)} JD chunks...") 
           embeddings=self.generate_embeddings(texts)


           # prepare metadata
           metadatas=[]
           for chunk in chunks:
               metadata=chunk["metadata"].copy()
               metadata["jd_id"] = jd_id
           if role:
               metadata["role"] = role
           if blob_url:
               metadata["blob_url"] = blob_url  # Store blob URL in metadata
           metadatas.append(metadata)


                                                

    #generate unique IDs for each chunk
           ids=[f"{jd_id}_chunk_{i}" for i in range(len(chunks))]

           # Store in ChromaDB (vector database - for fast semantic search)
           self.jd_collection.add(
                   embeddings=embeddings,
                   documents=texts,
                   metadatas=metadatas,
                   ids=ids
            )
           print(f"✓ Stored JD {jd_id} with {len(chunks)} chunks")
           return jd_id

    def store_cv(
            self,
            chunks:List[Dict],
            cv_id:Optional[str]=None,
            candidate_name:Optional[str]=None,
            original_fil_path:Optional[str]=None
    )->str:

        """
        Store CV chunks in vector database and original file in Azure Blob

        Args:
            chunks: List of chunk dictionaries
            cv_id: Optional CV identifier
            candidate_name: Candidate's name
            original_file_path: Path to original PDF/DOCX file

        Returns:
            cv_id for reference
        """

        if not cv_id:
            cv_id = str(uuid.uuid4())
       
         # Upload original file to Azure Blob Storage
        blob_url = None
        if original_file_path and self.container_client:
            blob_name = f"jds/{jd_id}/{os.path.basename(original_file_path)}"
            blob_url = self.upload_to_blob(original_file_path, blob_name)
        
        # Extract text from chunks
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} JD chunks...")
        embeddings = self.generate_embeddings(texts)
        
        # Prepare metadata
        metadatas = []
        for chunk in chunks:
            metadata = chunk["metadata"].copy()
            metadata["jd_id"] = jd_id
            if role:
                metadata["role"] = role
            if blob_url:
                metadata["blob_url"] = blob_url  # Store blob URL in metadata
            metadatas.append(metadata)
        
        # Generate unique IDs for each chunk
        ids = [f"{jd_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Store in ChromaDB (vector database - for fast semantic search)
        self.jd_collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✓ Stored JD {jd_id} with {len(chunks)} chunks")
        return jd_id
    
    def store_cv(
        self,
        chunks: List[Dict],
        cv_id: Optional[str] = None,
        candidate_name: Optional[str] = None,
        original_file_path: Optional[str] = None  # NEW parameter
    ) -> str:
        """
        Store CV chunks in vector database and original file in Azure Blob
        
        Args:
            chunks: List of chunk dictionaries
            cv_id: Optional CV identifier
            candidate_name: Candidate's name
            original_file_path: Path to original PDF/DOCX file
            
        Returns:
            cv_id for reference
        """
        if not cv_id:
            cv_id = str(uuid.uuid4())
        
        # Upload original file to Azure Blob Storage
        blob_url = None
        if original_file_path and self.container_client:
            blob_name = f"cvs/{cv_id}/{os.path.basename(original_file_path)}"
            blob_url = self.upload_to_blob(original_file_path, blob_name)
        
        # Extract text from chunks
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} CV chunks...")
        embeddings = self.generate_embeddings(texts)
        
        # Prepare metadata
        metadatas = []
        for chunk in chunks:
            metadata = chunk["metadata"].copy()
            metadata["cv_id"] = cv_id
            if candidate_name:
                metadata["candidate_name"] = candidate_name
            if blob_url:
                metadata["blob_url"] = blob_url  # Store blob URL
            metadatas.append(metadata)
        
        # Generate unique IDs
        ids = [f"{cv_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Store in ChromaDB (for vector search)
        self.cv_collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✓ Stored CV {cv_id} with {len(chunks)} chunks")
        return cv_id


    def generate_embeddings(
            self,
            texts: List[str])->List[List[float]]:
    
            # generate embeddings for all texts"
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()


    def query_jd(self, query: str, jd_id: Optional[str] = None, top_k: int = 5) -> Dict:
        """Search JD collection for relevant chunks"""
        query_embedding = self.generate_embeddings([query])[0]
        
        where = {"jd_id": jd_id} if jd_id else None
        
        results = self.jd_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        
        return results

    def query_cv(self, query: str, cv_id: Optional[str] = None, top_k: int = 5) -> Dict:
        """Search CV collection for relevant chunks"""
        query_embedding = self.generate_embeddings([query])[0]
        
        where = {"cv_id": cv_id} if cv_id else None
        
        results = self.cv_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        
        return results

    # test the vector db

if __name__ == "__main__":
    # Initialize vector DB
    vdb = VectorDBManager()

    # Example JD chunks
    test_chunks = [
        {"content": "Python developer with 5 years experience", "metadata": {"source": "test"}},
        {"content": "AWS and cloud computing expert", "metadata": {"source": "test"}}
    ]

    # Store JD and get jd_id
    jd_id = vdb.store_jd(test_chunks, role="Software Engineer")
    print(f"Stored JD: {jd_id}")  # ✅ jd_id exists here

    # Query JD
    results = vdb.query_jd("python programming", jd_id=jd_id)
    print(f"Query results: {results['documents']}")


                




               




                                                    













   
