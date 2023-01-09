from model import model_train


if __name__ == '__main__':
    use_exist_models = False
    if not use_exist_models:
        model_train("resnet18")