FROM python:3.11-slim

# ติดตั้ง system dependencies ที่จำเป็นสำหรับ OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ติดตั้ง Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดและโมเดลทั้งหมด
COPY . .

# Railway จะส่งตัวแปร $PORT มาให้ runtime
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
