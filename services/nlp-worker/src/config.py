from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 定義欄位與型別，Pydantic 會自動去讀環境變數
    # 例如：redis_host 會自動讀取環境變數 REDIS_HOST (大小寫不敏感)
    redis_host: str = "localhost" 
    redis_port: int = 6379
    redis_db: int = 0
    
    nlp_model_name: str = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

    class Config:
        # 告訴它去哪裡讀 .env 檔案
        env_file = ".env"
        env_file_encoding = 'utf-8'

# 實例化設定物件
settings = Settings()