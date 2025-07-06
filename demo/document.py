from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="data/InstructUIE.pdf",
    strategy="fast",
    languages=['eng'],
    skip_infer_table_types=True,
    infer_table_structure=False,
)
categories = [e.category for e in elements]
print(set(categories))