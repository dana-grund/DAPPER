import numpy as np

class NanCropper:
    """Wrap around the analysis to ensure failed samples are not considered."""

    def __init__(self, E):
        if not np.all(np.isfinite(E)):
            self.do_crop = True
            self.nan_samples = np.where(np.any(np.isnan(E), axis=1))[0]
            self.not_nan_samples = np.where(~np.any(np.isnan(E), axis=1))[0]
            print((
                "WARNING: Ensemble not finite. "
                f"Cropping {len(self.nan_samples)} samples: {self.nan_samples}"
            ))
        else:
            self.do_crop = False

    def crop(self, E):
        if self.do_crop:
            A = E[self.not_nan_samples, :]
            return A
        else:
            return E

    def uncrop(self, E):
        if self.do_crop:
            A = E
            if len(E.shape) > 1:
                # e.g., an ensemble
                nans = np.empty((E.shape[1],))
                nans[:] = np.nan
            else:
                # e.g., a scalar statistic per sample
                nans = np.nan
            for i in self.nan_samples:
                A = np.insert(A, i, nans, axis=0)
            return A
        else:
            return E

def crop_nans(E):
    """Single-use cropping."""
    nan_cropper = NanCropper(E)
    A = nan_cropper.crop(E)
    return A
