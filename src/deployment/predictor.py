"""
AOI Predictor

Runs semantic segmentation on an entire GeoTIFF by tiling it.
"""

from __future__ import annotations

from logging import config
from pathlib import Path

import rasterio
import numpy as np
from sklearn import base
import torch
from PIL import Image

from src.deployment.tiler import AOITiler
from src.training.inference.contracts import InferenceConfig
from src.training.inference.loader import CheckpointLoader
from src.training.models.factory import ModelFactory
from src.visualization import comparison, overlay


class AOIPredictor:

    def __init__(
        self,
        config,
        # model,
        checkpoint=None,
    ):
        
        self._config = config
        # self._model = model

        self._device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        #
        # Build exactly the same model used during training.
        #
        print("=" * 60)
        # print("MODEL INPUT CHANNELS:", config.model.in_channels)
        # print("ENABLED INDICES:", config.features.enabled_indices)
        print(type(config))
        print(config)
        print("=" * 60)  

        model_result = ModelFactory.build(config)
        self._model = model_result.model

        #
        # Restore checkpoint.
        #
        from src.training.inference import InferenceConfig
        from src.training.inference.loader import CheckpointLoader

        inf_cfg = InferenceConfig.from_config(config)

        loader = CheckpointLoader(inf_cfg)

        ckpt_path = loader.resolve_path()
        payload = loader.load(ckpt_path)
        loader.restore_model(self._model, payload)

        self._model.to(self._device)
        self._model.eval()

        self._patch_size = (
            config.inference.patch_size
        )

        self._stride = (
            config.inference.stride
        )

    def predict_scene(
        self,
        raster_path,
    ):
        """
        Generator.

        Returns one prediction tile at a time.
        """
        print("Patch size:", self._patch_size)
        print("Stride:", self._stride)
        
        tiler = AOITiler(
            patch_size=self._patch_size,
            stride=self._stride,
        )

        with torch.no_grad():

            for tile in tiler.generate(
                raster_path,
            ):

                image = (
                    torch.from_numpy(tile.image)
                    .unsqueeze(0)
                    .float()
                    .to(self._device)
                )

                logits = self._model(image)

                #
                # Some architectures return tuple/list.
                #
                if isinstance(
                    logits,
                    (tuple, list),
                ):
                    logits = logits[0]

                prediction = (
                    logits.argmax(1)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                yield {
                    "mask": prediction,
                    "row": tile.row,
                    "col": tile.col,
                    "height": tile.height,
                    "width": tile.width,
                }
    
    def predict_mosaic(
        self,
        raster_path,
    ):
        """
        Predict an entire AOI and stitch all prediction
        tiles into one segmentation mask.
        """

        import rasterio

        with rasterio.open(raster_path) as src:
            # print("Band count:", src.count)

            # for i in [1, 2, 3]:
            #     band = src.read(i)

            #     print(f"Band {i}")
            #     print("dtype:", band.dtype)
            #     print("shape:", band.shape)
            #     print("nodata:", src.nodata)
            #     print("NaNs:", np.isnan(band).sum())
            #     print("min:", np.nanmin(band))
            #     print("max:", np.nanmax(band))

            height = src.height
            width = src.width

        mosaic = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        for tile in self.predict_scene(raster_path):

            r = tile["row"]
            c = tile["col"]

            h = tile["height"]
            w = tile["width"]

            mosaic[
                r:r+h,
                c:c+w,
            ] = tile["mask"][:h, :w]

        return mosaic

    def save_prediction(
        self,
        raster_path,
        output_png,
    ):
        """
        Run inference on an entire AOI and save:

        *_mask.png
        *_prediction_color.png
        *_rgb.png
        *_overlay.png
        *_comparison.png
        """

        # ------------------------------------------------------------------
        # Predict complete AOI
        # ------------------------------------------------------------------
        mask = self.predict_mosaic(raster_path)

        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)

        base = output_png.with_suffix("")

        # ------------------------------------------------------------------
        # Read RGB from GeoTIFF
        # ------------------------------------------------------------------
        with rasterio.open(raster_path) as src:

            # ---------- DEBUG ----------
            print("\n========== TIFF CHECK ==========")

            for i in range(1, src.count + 1):
                band = src.read(i)

                valid = np.count_nonzero(~np.isnan(band))
                total = band.size

                print(
                    f"Band {i:2d}",
                    "Valid:", valid,
                    "Total:", total,
                    "Percent:", round(valid / total * 100, 2),
                )

            print("================================\n")
            # ---------- END DEBUG ----------

            print("Band count:", src.count)

            for i in [1, 2, 3]:
                band = src.read(i)

                print(f"Band {i}")
                print("dtype:", band.dtype)
                print("shape:", band.shape)
                print("nodata:", src.nodata)
                print("All NaN:", np.isnan(band).all())
                print("NaNs:", np.isnan(band).sum())
                print("min:", np.nanmin(band))
                print("max:", np.nanmax(band))

            # Sentinel-2 harmonized order:
            # Blue Green Red NIR SWIR1 SWIR2 ...
            rgb = src.read([3, 2, 1]).transpose(1, 2, 0).astype(np.float32)

        rgb = np.nan_to_num(rgb, nan=0.0)

        mn = rgb.min()
        mx = rgb.max()

        if mx > mn:
            rgb = (rgb - mn) / (mx - mn)
        else:
            rgb[:] = 0

        # # Stretch to 0-255
        # rgb -= rgb.min()

        # if rgb.max() > 0:
        #     rgb /= rgb.max()

        print(np.nanmin(rgb), np.nanmax(rgb))
        print(np.isnan(rgb).sum())
        print(rgb.dtype)

        rgb = (rgb * 255).astype(np.uint8)

        # ------------------------------------------------------------------
        # Color palette
        # ------------------------------------------------------------------
        palette = np.array(
            [
                [0, 0, 0],          # background
                [0, 0, 255],        # water
                [255, 220, 0],      # sand
                [0, 180, 0],        # vegetation
            ],
            dtype=np.uint8,
        )

        color_mask = palette[mask]

        # ------------------------------------------------------------------
        # Overlay
        # ------------------------------------------------------------------
        overlay = (
            0.6 * rgb.astype(np.float32)
            + 0.4 * color_mask.astype(np.float32)
        ).astype(np.uint8)

        # ------------------------------------------------------------------
        # Side-by-side comparison
        # ------------------------------------------------------------------
        comparison = np.concatenate(
            (
                rgb,
                color_mask,
                overlay,
            ),
            axis=1,
        )

        # ------------------------------------------------------------------
        # Save outputs
        # ------------------------------------------------------------------
        Image.fromarray(mask).save(base.with_name(base.name + "_mask.png"))

        Image.fromarray(color_mask).save(
            base.with_name(base.name + "_prediction_color.png")
        )

        Image.fromarray(rgb).save(
            base.with_name(base.name + "_rgb.png")
        )

        Image.fromarray(overlay).save(
            base.with_name(base.name + "_overlay.png")
        )

        Image.fromarray(comparison).save(
            base.with_name(base.name + "_comparison.png")
        )

        return {
            "mask": str(base.with_name(base.name + "_mask.png")),
            "rgb": str(base.with_name(base.name + "_rgb.png")),
            "prediction": str(base.with_name(base.name + "_prediction_color.png")),
            "overlay": str(base.with_name(base.name + "_overlay.png")),
            "comparison": str(base.with_name(base.name + "_comparison.png")),
        }    