# Guide to testing

chat_model -> tests all chat models
image_model -> tests all image models
test_user -> tests a single model of your choice

# Run all tests
Run: `pytest`


# To test chat models

Run: `python -m pytest testLib/test_chat_model.py -v`


# To test image models

Run: `python -m pytest testLib/test_image.py -v`

# To test a single model

Run: `python -m testLib.test_user model_name`
