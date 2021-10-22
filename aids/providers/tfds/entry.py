import tensorflow_datasets as tfds

ds, info = tfds.load(
            name='glue/sst2:2.0.0',
            split='validation',
            shuffle_files=False,
            with_info=True,
            as_supervised=False,
)

df = tfds.as_dataframe(
    ds, info
)

print(df.head(10))
