.PHONY: setup run dashboard api test notebooks
setup:      ; pip install -r requirements.txt
run:        ; python run_pipeline.py
dashboard:  ; streamlit run app/dashboard.py
api:        ; uvicorn service.main:app --reload --port 8000
test:       ; python -m pytest -q tests
notebooks:  ; jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
