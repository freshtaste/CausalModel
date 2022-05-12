.PHONY: demo test

demo:
	pipenv run python3 demo.py

simplified:
	pipenv run python3 simplified.py

test:
	pipenv run pytest --doctest-modules
