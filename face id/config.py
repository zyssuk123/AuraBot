"""
Configuration Blindy AI - Ajuste ces paramètres selon ton environnement
"""

# === PARAMÈTRES MICRO / ÉCOUTE ===
DUREE_ECOUTE = 4.0           # Secondes pour écouter une question (3-6 recommandé)
DUREE_ACTIVATION = 2.0       # Secondes pour détecter "Blindy" (1.5-3 recommandé)
SEUIL_SILENCE = 0.01         # Sensibilité au bruit (0.005-0.02, baisse si micro trop sensible)

# === PARAMÈTRES RECONNAISSANCE VOCALE ===
MOTS_MAGIQUES = ["blindy", "blende", "blind", "blynde", "blandy", "bindy"]
LANGUE = "fr-FR"             # fr-FR, en-US, etc.

# === PARAMÈTRES GEMINI IA ===
GEMINI_TIMEOUT = 8           # Secondes d'attente max pour réponse IA (5-10)
GEMINI_MODELE = "gemini-2.0-flash"  # gemini-1.5-flash ou gemini-2.0-flash
MAX_CHARS_REPONSE = 250      # Limite caractères réponse (150-300)

# === PARAMÈTRES SYNTHÈSE VOCALE ===
VITESSE_PAROLE_COURT = 155   # Mots/min pour phrases courtes
VITESSE_PAROLE_LONG = 170    # Mots/min pour phrases longues
VOLUME = 0.9                 # 0.0 à 1.0

# === PARAMÈTRES ANTI-SPAM ===
DELAI_ANNONCE_NOM = 15       # Secondes entre 2 annonces du même nom

# === DEBUG ===
AFFICHER_LOGS = True         # Affiche les logs dans la console
