import pyarrow.parquet as pq


class WikiTextDataset:
    def __init__(self, file_path):
        # Load the .parquet file using pyarrow
        table = pq.read_table(file_path)
        print("Schema:", table.schema)

        # Extract the 'text' column directly
        try:
            self.texts = table['text'].to_pylist()
        except KeyError:
            raise ValueError("The 'text' column is missing in the .parquet file.")

        # Debug: Print the first few entries
        print("Loaded Data:")
        print(self.texts[:5])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


# Example usage
if __name__ == "__main__":
    file_path = "external/wikitext/wikitext-2-v1/train-00000-of-00001.parquet"
    dataset = WikiTextDataset(file_path)

    # Print the first 5 entries
    for i in range(5):
        print(dataset[i])
