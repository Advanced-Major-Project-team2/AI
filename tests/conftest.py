# tests/conftest.py
# 역할:
#   - pytest 시작 시 프로젝트 루트의 .env를 읽어, os.environ에 주입합니다.
#   - 이렇게 하면 _env_ready()가 os.getenv로 읽을 때 값이 보입니다.

from dotenv import load_dotenv

# 디폴트 경로(.env)를 자동 탐색하여 로드
load_dotenv()
