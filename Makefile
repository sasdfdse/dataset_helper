APP_DIR  := augmentor
VENV     := $(APP_DIR)/.venv
PYTHON   ?= python3

# Follow the installed venv kind (small/large); default to large.
VENV_KIND ?= $(if $(wildcard $(VENV)/.installed-large),large,$(if $(wildcard $(VENV)/.installed-small),small,large))
VENV_STAMP := $(VENV)/.installed-$(VENV_KIND)

.DEFAULT_GOAL := help
.PHONY: augmentor augmentator venv venv-small venv-large test test-gpu deps help

# The GUI runs its worker with its own Python, so always launch with the venv's Python.
augmentor: $(VENV_STAMP) ## 데이터셋 도구 GUI 실행
	cd $(APP_DIR) && .venv/bin/python main.py

augmentator: augmentor

venv: venv-large ## Python 가상환경 + Large 의존성 (8GB GPU 기본 모델, 빠른 모드 포함)

venv-large: $(VENV)/.installed-large

venv-small: $(VENV)/.installed-small ## Python 가상환경 + Small 의존성 (작은 모델용, 4bit/빠른 모드 제외)

$(VENV)/.installed-%: $(APP_DIR)/requirements.txt $(APP_DIR)/requirements-bg-small.txt $(APP_DIR)/requirements-bg-large.txt
	@test -x $(VENV)/bin/python || $(PYTHON) -m venv --system-site-packages $(VENV)
	$(VENV)/bin/pip install -r $(APP_DIR)/requirements.txt -r $(APP_DIR)/requirements-bg-$*.txt
	@touch $@

test: $(VENV_STAMP) ## 테스트 실행 (GPU 테스트 제외)
	cd $(APP_DIR) && .venv/bin/python -m pytest -q

test-gpu: $(VENV_STAMP) ## GPU 테스트 실행 (CUDA와 모델 필요)
	cd $(APP_DIR) && .venv/bin/python -m pytest -m gpu -q tests/test_bg_gpu_smoke.py

deps: ## 시스템 패키지 설치 (sudo 필요, Ubuntu/Debian). Video Tool 탭은 ffmpeg를 쓴다
	sudo apt-get update -qq
	sudo apt-get install -y python3-venv python3-tk ffmpeg

help: ## 명령 목록
	@grep -hE '^[a-z_-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/^([a-z_-]+):.*## /\1\t/' | awk -F'\t' '{printf "  make %-12s %s\n", $$1, $$2}'
