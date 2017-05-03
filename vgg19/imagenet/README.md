# Download ImageNet

These are scripts for downloading ImageNet.

# Usage

## I. Download the image urls.

```
$ python get_urls.py
```

## II. Create the database.

```
$ python create_db.py
```

## III. Download the images.

```
$ python download_images.py
```

- It takes a long time, but you can resume downloading.
- Change the parameters (n_threads and time.sleep()) as needed.

## IV. Make the npy files.

```
$ python preprocess.py
```

