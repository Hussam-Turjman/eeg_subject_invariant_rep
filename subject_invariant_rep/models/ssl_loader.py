from .utils import to_logits


def create_ssl_loaders(loader, fold, transform, batch_size, drop_last, num_class, n_subjects, EEGSubsetClass,
                       FixedDatasetClass):
    X_train, y_train, subject_ids_train = loader.load_train_set(fold=fold, load_subject_ids=True)
    X_val, y_val, subject_ids_val = loader.load_val_set(fold=fold, load_subject_ids=True)
    X_test, y_test, subject_ids_test = loader.load_test_set(fold=fold, load_subject_ids=True)

    y_train = to_logits(y_train, num_class)
    y_val = to_logits(y_val, num_class)
    y_test = to_logits(y_test, num_class)

    subject_ids_train = to_logits(subject_ids_train, n_subjects)
    subject_ids_val = to_logits(subject_ids_val, n_subjects)
    subject_ids_test = to_logits(subject_ids_test, n_subjects)

    train_set = EEGSubsetClass(X_train, y_train, subject_ids_train, transform=transform)
    val_set = EEGSubsetClass(X_val, y_val, subject_ids_val, transform=transform)
    test_set = EEGSubsetClass(X_test, y_test, subject_ids_test, transform=None)

    train_set = FixedDatasetClass(train_set)
    val_set = FixedDatasetClass(val_set)

    train_loader = train_set.dataloader(
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=2
    )
    val_loader = val_set.dataloader(
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=2
    )
    test_loader = test_set.dataloader(
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=2
    )
    return train_loader, val_loader, test_loader


__all__ = ["create_ssl_loaders"]
