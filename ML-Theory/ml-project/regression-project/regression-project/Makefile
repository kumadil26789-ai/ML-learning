.PHONY: install data train test serve docker
install:
	pip install -r requirements.txt
data:
	python -m regml.generate_data
train:
	python -m regml.train
test:
	python -m unittest discover -s tests -t . -v
serve:
	gunicorn --bind 0.0.0.0:8000 "regml.api:create_app()"
docker:
	docker build -t house-price-api . && docker run -p 8000:8000 house-price-api
