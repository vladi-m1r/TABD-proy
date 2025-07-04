import argparse
from server.store_csv import store_csv_in_chroma
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="Ruta al archivo CSV local")
    args = parser.parse_args()
    csv_path = args.csv_path
    if not os.path.isfile(csv_path):
        print(f"El archivo {csv_path} no existe.")
        return
    store_csv_in_chroma(csv_path)
    print(f"CSV {csv_path} almacenado en ChromaDB correctamente.")

if __name__ == "__main__":
    main()
