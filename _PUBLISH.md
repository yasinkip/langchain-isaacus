To publish a new version:
```bash
# Run tests
poetry run pytest --disable-socket --allow-unix-socket --asyncio-mode=auto tests/unit_tests
poetry run pytest --asyncio-mode=auto tests/integration_tests

# Set PyPi token
poetry config pypi-token.pypi YOUR_PYPI_TOKEN_HERE

# Publish
poetry build
poetry publish
```