.PHONY: all homogeneous heterogeneous incomplete more_groups test

all: homogeneous heterogeneous incomplete more_groups

homogeneous:
	pipenv run python3 example.py --config_name homogeneous

heterogeneous:
	pipenv run python3 example.py --config_name heterogeneous

incomplete:
	pipenv run python3 example.py --config_name incomplete

more_groups:
	pipenv run python3 example.py --config_name more_groups

test:
	pipenv run pytest --doctest-modules
