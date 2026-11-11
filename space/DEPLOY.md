# Deploying the demo to Hugging Face Spaces

The web demo is the Gradio app in this folder. It installs the `cartoonify`
package straight from this GitHub repository, so the demo always runs the
latest code on `main`.

## One-time setup (about 5 minutes)

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Go to **New Space**: https://huggingface.co/new-space
3. Fill in:
   - **Space name**: `cartoonify` (the public URL becomes
     `https://<your-username>-cartoonify.hf.space`)
   - **SDK**: Gradio
   - **License**: MIT
   - **Visibility**: Public
4. Click **Create Space**.

## Upload the files

On the Space page, open the **Files** tab → **Add file** → **Upload files**,
and upload the contents of this folder:

```
app.py
requirements.txt
README.md
examples/input.jpg
```

(`README.md` here is the Space configuration file with the front-matter block —
it replaces the default one the Space created. Upload it too.)

The Space builds automatically. When the status turns from *Building* to
*Running*, the demo is live at `https://<your-username>-cartoonify.hf.space`.

## Notes

- `requirements.txt` installs the package from GitHub
  (`cartoonify @ git+https://github.com/fari-cloud/Cartoonify-in-Computer-vision`).
  Merge the project PR to `main` first, otherwise the build fails because the
  package does not exist on `main` yet.
- To update the demo later, edit the files on the Space (or connect the Space
  to a GitHub repo under **Settings**) and it rebuilds automatically.
