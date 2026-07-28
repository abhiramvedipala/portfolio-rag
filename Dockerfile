FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ data/
COPY scripts/ scripts/
COPY llm/ llm/
COPY api/ api/

# chroma_db/ is gitignored, so it isn't in this build context -- build it
# fresh from data/about_abhiram.md as part of the image build.
RUN python scripts/embed.py

# HF Spaces (Docker SDK) routes traffic to port 7860 specifically.
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
