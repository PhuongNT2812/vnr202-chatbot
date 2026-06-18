import importlib
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import firebase_admin
from firebase_admin import credentials, firestore

# Tải API Key từ file .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def process_and_upload_data():
    print("🚀 Bắt đầu quá trình xử lý dữ liệu RAG...")

    # BƯỚC 1: LOADER (Đọc tất cả file trong thư mục data_raw)
    folder_path = "data_raw"
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Thư mục {folder_path} không tồn tại")

    def _import_loader(module_name, class_name):
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            return None

    Docx2txtLoader = _import_loader("langchain_community.document_loaders", "Docx2txtLoader")
    UnstructuredWordDocumentLoader = _import_loader("langchain.document_loaders", "UnstructuredWordDocumentLoader")

    documents = []
    supported_extensions = [".pdf", ".doc", ".docx", ".txt", ".md"]
    raw_files = sorted(
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in supported_extensions
    )

    if not raw_files:
        raise FileNotFoundError(f"Không tìm thấy file phù hợp trong thư mục {folder_path}")

    print(f"1. Đang đọc {len(raw_files)} file từ: {folder_path}")
    for file_name in raw_files:
        file_path = os.path.join(folder_path, file_name)
        ext = os.path.splitext(file_name)[1].lower()
        print(f"   - Đang đọc file: {file_path}")

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".docx", ".doc"]:
            if Docx2txtLoader is not None and ext == ".docx":
                loader = Docx2txtLoader(file_path)
            elif UnstructuredWordDocumentLoader is not None:
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                raise RuntimeError(
                    "Chưa cài đặt loader Word phù hợp. Hãy cài thư viện hỗ trợ tải .doc/.docx hoặc kiểm tra lại phiên bản LangChain."
                )
        elif ext in [".txt", ".md"]:
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            print(f"   - Bỏ qua định dạng không hỗ trợ: {file_path}")
            continue

        documents.extend(loader.load())

    # BƯỚC 2: CHUNKING (Cắt nhỏ văn bản)
    # Cắt thành các đoạn 1000 ký tự, phần giao nhau (overlap) là 200 ký tự để không mất ngữ cảnh
    print("2. Đang băm nhỏ tài liệu (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   -> Đã cắt thành {len(chunks)} đoạn nhỏ.")

    # BƯỚC 3: EMBEDDING (Nhúng dữ liệu thành Vector)
    print("3. Đang khởi tạo mô hình Embedding của Gemini...")
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # BƯỚC 4: VECTOR STORE (Đẩy lên Firestore Database)
    print("4. Đang kết nối với Firebase và đẩy dữ liệu lên mây...")
    # Khởi tạo Firebase (Lưu ý: Bạn sẽ cần tải file Service Account JSON từ Firebase về để xác thực thực tế)
    if not firebase_admin._apps:
        # Tạm thời dùng thông tin mặc định nếu bạn đã login firebase CLI
        cred = credentials.ApplicationDefault() 
        firebase_admin.initialize_app(cred, {
            'projectId': 'vnr202-chatbot',
        })
    
    db = firestore.client()
    collection_ref = db.collection('lich_su_dang_vectors')

    # Xử lý từng đoạn chunk, biến thành vector và lưu lên Firestore
    for i, chunk in enumerate(chunks):
        # Tạo vector cho đoạn text
        vector = embeddings_model.embed_query(chunk.page_content)
        
        # Đóng gói dữ liệu để đưa lên database
        doc_data = {
            "text": chunk.page_content,
            "embedding": vector, # Dãy số vector
            "metadata": chunk.metadata # Số trang, tên file...
        }
        
        # Lưu vào collection 'lich_su_dang_vectors'
        collection_ref.add(doc_data)
        print(f"   -> Đã tải lên đoạn {i+1}/{len(chunks)}")

    print("✅ Hoàn tất! Dữ liệu đã sẵn sàng trên Firebase.")

if __name__ == "__main__":
    process_data()