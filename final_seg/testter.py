from our_helper_config import Configuration
from JTMLFormer import HighResolutionNet
import os
import sys





def main():
    config = Configuration()

    model = HighResolutionNet(config)

    print(config.loss["params"]["ce_ignore_index"])

if __name__ == "__main__":
    main()