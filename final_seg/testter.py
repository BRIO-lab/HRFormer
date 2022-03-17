from our_helper_config import Configuration

def main():
    config = Configuration()

    print(config.loss["params"]["ce_ignore_index"])

if __name__ == "__main__":
    main()