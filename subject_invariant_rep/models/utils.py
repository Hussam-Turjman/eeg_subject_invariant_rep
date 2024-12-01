import math

import numpy as np


def _early_stopping(losses, patience, tolerance):
    losses = np.array(losses)

    if patience is None:
        return False

    if len(losses) < patience:
        return False

    train_loss = np.flip(losses)

    a = train_loss[0]
    for b in train_loss[1:patience]:
        if not math.isclose(a, b, rel_tol=tolerance):
            return False

    return True


def early_stop(*losses, patience=3, tolerance=1e-3, check_all=False) -> bool:
    res = []
    for loss in losses:
        do_stop = _early_stopping(loss, patience, tolerance)
        res.append(do_stop)
    if check_all:
        return all(res)
    return any(res)


def make_loader_fixed(loader):
    batches = []
    for batch in loader:
        batches.append(batch)
    return batches


def to_logits(y: np.ndarray, num_classes):
    y = y.astype(np.int32)
    logits = np.zeros((y.shape[0], num_classes))
    logits[np.arange(y.shape[0]), y] = 1.0
    return logits


def to_labels(y: np.ndarray):
    return np.argmax(y, axis=-1)


def set_freeze(model, freeze=True):
    for param in model.parameters():
        param.requires_grad = not freeze


def evaluate_ssl(loader, encoder, downstream, subject_identifier):
    y_pred = []
    y_true = []
    embeddings = []
    ids_true = []
    ids_pred = []
    encoder.eval()
    downstream.eval()

    set_freeze(encoder, freeze=True)
    set_freeze(downstream, freeze=True)
    if subject_identifier is not None:
        subject_identifier.eval()
        set_freeze(subject_identifier, freeze=True)
    for batch in loader:
        X, y, subjects_ids = batch
        if type(X) is list:
            X = X[0]

        latent = encoder(X)
        if subject_identifier is not None:
            identity = subject_identifier(latent)
        else:
            identity = None
        logits = downstream(latent)
        logits = logits.cpu().detach().numpy()
        y = y.cpu().numpy()
        y = y.argmax(axis=1)
        logits = logits.argmax(axis=1)
        latent = latent.cpu().detach().numpy()
        y_pred.extend(logits)
        y_true.extend(y)
        embeddings.append(latent)
        ids_true.extend(subjects_ids.cpu().numpy().argmax(axis=1))
        if identity is not None:
            ids_pred.extend(identity.cpu().numpy().argmax(axis=1))
        else:
            ids_pred.append([None]*len(subjects_ids))

    embeddings = np.concatenate(embeddings, axis=0)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    return y_true, y_pred, embeddings, ids_true, ids_pred


__all__ = ["early_stop", "make_loader_fixed", "to_logits", "to_labels", "set_freeze", "evaluate_ssl"]
