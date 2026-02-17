from transformers import pipeline
from config import settings
import time

# [關鍵優化]：模型載入是非常耗時的 (可能要 5-10 秒)
# 所以我們要在全域 (Global) 載入，讓它只在 Worker 啟動時跑一次。
# 絕對不要寫在 function 裡面，否則每處理一個任務都要重新載入模型。
print(f"正在載入模型: {settings.nlp_model_name} ...")

# 建立一個 Sentiment Analysis Pipeline
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=settings.nlp_model_name,
        device=-1  # -1 代表使用 CPU，如果你有 GPU 可以設為 0
    )
    print("模型載入完成！")
except Exception as e:
    print(f"模型載入失敗: {e}")
    sentiment_pipeline = None

def perform_heavy_nlp(text: str) -> dict:
    """
    使用 Hugging Face 模型進行情緒分析
    """
    if not sentiment_pipeline:
        return {"error": "Model not loaded"}

    print(f"   [AI Logic] 分析中: {text[:30]}...")
    start_time = time.time()

    # 1. 執行預測 (真正的 AI 運算)
    # output 格式範例: [{'label': 'positive', 'score': 0.95}]
    try:
        # 限制長度避免爆記憶體，並截斷過長文字
        results = sentiment_pipeline(text, truncation=True, max_length=512)
        top_result = results[0]
        
        # 將標籤標準化 (不同模型標籤可能不同)
        label_map = {
            "positive": "利多 (Positive)", 
            "negative": "利空 (Negative)", 
            "neutral": "中立 (Neutral)"
        }
        raw_label = top_result['label']
        mapped_label = label_map.get(raw_label, raw_label)

    except Exception as e:
        return {"error": str(e)}

    process_time = time.time() - start_time

    return {
        "original_text_snippet": text[:50],
        "sentiment": mapped_label,
        "confidence_score": round(top_result['score'], 4),
        "model_used": settings.nlp_model_name,
        "process_time_seconds": round(process_time, 3)
    }