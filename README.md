# TinyLLM (app)

The Android side of TinyLLM: loads `app/src/main/assets/model.bin` and
runs a hand-written transformer (`Transformer.kt`) to generate text from
whatever you type, entirely on-device. No PyTorch/TFLite/ONNX here --
just plain numbers and array math.

This repo has no Python and does no training. The model is trained in a
separate repo (see [greafstacks/TinyLLM-trainer](../trainer), or whatever
you name it) and `model.bin` is committed here as a normal binary asset,
the same way you've already been swapping `corpus.txt` in and out. To
update the model: download the `model-bin` artifact from the trainer
repo's latest Actions run, and replace
`app/src/main/assets/model.bin` with it.

## Getting an APK

Every push to `main` builds a debug APK via GitHub Actions (no local
Android Studio needed): Actions tab -> latest **Build APK** run ->
download `app-debug-apk` (a zip containing `app-debug.apk`) -> install
on your phone. This build does not train anything, so it only takes a
couple of minutes.

## Notes

- If `model.bin`'s format ever changes (a new `TLM` version), Kotlin will
  refuse to load it with a clear error rather than silently misreading
  it -- check the version prefix in `Transformer.kt` against what the
  trainer repo's `model_ref.py` wrote.
- Debug build (unsigned): fine for your own/friends' phones.
- minSdk 24 (Android 7.0+).
