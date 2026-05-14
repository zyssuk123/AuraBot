"""
cloud_gallery.py — Gestionnaire de galerie biométrique sur Vercel Blob
=======================================================================
Remplace le stockage local `galerie/` par Vercel Blob (gratuit jusqu'à 1 Go).

Flux :
  - Inscription  → upload de la photo vers Vercel Blob
  - Entraînement → téléchargement temporaire des photos pour LBPH
  - Suppression  → via l'API Blob

Configuration .env requise :
  BLOB_READ_WRITE_TOKEN=vercel_blob_rw_xxxx...
"""

import os
import io
import tempfile
import requests
from dotenv import load_dotenv

load_dotenv()

BLOB_TOKEN = os.getenv("BLOB_READ_WRITE_TOKEN")
BLOB_API   = "https://blob.vercel-storage.com"

# Préfixe pour isoler les fichiers Blindy dans le bucket
BLOB_PREFIX = "blindy-galerie/"


def _headers() -> dict:
    """Headers d'authentification Vercel Blob."""
    if not BLOB_TOKEN:
        raise RuntimeError(
            "BLOB_READ_WRITE_TOKEN manquant dans .env\n"
            "Obtiens-le sur : https://vercel.com/dashboard → Storage → Blob"
        )
    return {
        "authorization": f"Bearer {BLOB_TOKEN}",
        "x-api-version":  "7",
    }


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

def upload_photo(person_name: str, photo_index: int, image_bytes: bytes) -> str:
    """
    Uploade une photo JPEG vers Vercel Blob.
    
    Args:
        person_name  : nom de la personne (ex: "Abdel")
        photo_index  : numéro de la photo (1, 2, ...)
        image_bytes  : contenu JPEG brut
    
    Returns:
        URL publique de la photo sur Vercel Blob
    """
    safe_name = person_name.replace(" ", "_").capitalize()
    blob_path = f"{BLOB_PREFIX}{safe_name}/{safe_name}_{photo_index}.jpg"

    response = requests.put(
        f"{BLOB_API}/{blob_path}",
        headers={
            **_headers(),
            "content-type": "image/jpeg",
            "x-add-random-suffix": "0",   # On garde le chemin exact
        },
        data=image_bytes,
        timeout=30,
    )

    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"[Blob] Upload échoué ({response.status_code}): {response.text}"
        )

    url = response.json().get("url", "")
    print(f"[Blob] ✅ Photo uploadée : {url}")
    return url


# ─────────────────────────────────────────────────────────────────────────────
# LIST
# ─────────────────────────────────────────────────────────────────────────────

def list_photos(person_name: str | None = None) -> list[dict]:
    """
    Liste les photos dans Vercel Blob.
    
    Args:
        person_name : si fourni, filtre par personne ; sinon retourne tout
    
    Returns:
        Liste de dicts avec 'url', 'pathname', 'size'
    """
    prefix = BLOB_PREFIX
    if person_name:
        safe_name = person_name.replace(" ", "_").capitalize()
        prefix = f"{BLOB_PREFIX}{safe_name}/"

    response = requests.get(
        f"{BLOB_API}",
        headers=_headers(),
        params={"prefix": prefix, "limit": 1000},
        timeout=20,
    )

    if response.status_code != 200:
        print(f"[Blob] Erreur list ({response.status_code}): {response.text}")
        return []

    return response.json().get("blobs", [])


def list_persons() -> list[str]:
    """
    Retourne la liste des personnes enregistrées dans Vercel Blob.
    Les noms correspondent aux sous-dossiers du préfixe Blindy.
    """
    blobs = list_photos()
    noms = set()
    for b in blobs:
        path = b.get("pathname", "")
        # ex: blindy-galerie/Abdel/Abdel_1.jpg → "Abdel"
        parts = path.replace(BLOB_PREFIX, "").split("/")
        if len(parts) >= 2:
            noms.add(parts[0].replace("_", " "))
    return sorted(noms)


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD (pour entraînement local temporaire)
# ─────────────────────────────────────────────────────────────────────────────

def download_gallery_to_tempdir() -> str:
    """
    Télécharge toute la galerie Blob dans un dossier temporaire.
    Reproduit la structure locale : temp/NomPersonne/photo.jpg
    
    Returns:
        Chemin du dossier temporaire (à utiliser pour train_on_gallery)
    """
    blobs = list_photos()
    if not blobs:
        print("[Blob] Galerie vide — aucune photo à télécharger.")
        tmp_dir = tempfile.mkdtemp(prefix="blindy_galerie_")
        return tmp_dir

    tmp_dir = tempfile.mkdtemp(prefix="blindy_galerie_")
    print(f"[Blob] Téléchargement de {len(blobs)} photos dans {tmp_dir}...")

    for blob in blobs:
        url      = blob.get("url", "")
        pathname = blob.get("pathname", "")
        # ex: blindy-galerie/Abdel/Abdel_1.jpg
        relative = pathname.replace(BLOB_PREFIX, "")  # Abdel/Abdel_1.jpg
        local_path = os.path.join(tmp_dir, relative.replace("/", os.sep))

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(r.content)
            else:
                print(f"[Blob] Erreur download {url}: {r.status_code}")
        except Exception as e:
            print(f"[Blob] Erreur réseau {url}: {e}")

    print(f"[Blob] ✅ Galerie prête dans {tmp_dir}")
    return tmp_dir


# ─────────────────────────────────────────────────────────────────────────────
# DELETE
# ─────────────────────────────────────────────────────────────────────────────

def delete_person(person_name: str) -> int:
    """
    Supprime toutes les photos d'une personne dans Vercel Blob.
    
    Returns:
        Nombre de fichiers supprimés
    """
    blobs  = list_photos(person_name)
    urls   = [b["url"] for b in blobs]
    if not urls:
        print(f"[Blob] Aucune photo trouvée pour {person_name}")
        return 0

    response = requests.delete(
        f"{BLOB_API}",
        headers={**_headers(), "content-type": "application/json"},
        json={"urls": urls},
        timeout=20,
    )

    if response.status_code == 200:
        print(f"[Blob] ✅ {len(urls)} photos supprimées pour {person_name}")
        return len(urls)
    else:
        print(f"[Blob] Erreur suppression ({response.status_code}): {response.text}")
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# TEST RAPIDE (python cloud_gallery.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== TEST Vercel Blob ===")
    try:
        personnes = list_persons()
        print(f"Personnes enregistrées : {personnes if personnes else '(vide)'}")

        tmp = download_gallery_to_tempdir()
        print(f"Galerie temporaire : {tmp}")
    except RuntimeError as e:
        print(f"⚠️  {e}")
