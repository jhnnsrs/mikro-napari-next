from mikro_napari.run import main

from rich.logging import RichHandler
import logging

logging.basicConfig(level="INFO", handlers=[RichHandler()])

if __name__ == "__main__":
    main()
