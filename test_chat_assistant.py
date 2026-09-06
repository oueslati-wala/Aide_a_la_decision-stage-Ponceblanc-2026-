import chat_assistant


def test_detects_price_questions_with_facturer_variants():
    assert chat_assistant._looks_like_price_request("combien faut-il facturer ?")
    assert chat_assistant._looks_like_price_request("donne-moi le prix recommandé")
    assert chat_assistant._looks_like_price_request("quelle est la recommandation ?")


def test_direct_price_shortcut_uses_interface_context():
    context = {
        "source": "ponceblanc",
        "client": "GERFLOR",
        "produit": "LIASSE",
        "quantite": 1000,
        "cout_total": 6000.0,
        "pricing_mode": "balanced",
        "low_acceptance_percentile": 25.0,
    }

    reply = chat_assistant._try_direct_price_recommendation(
        "Combien faut-il facturer ?",
        source_hint="ponceblanc",
        form_context=context,
    )

    assert reply is not None
    assert "Prix recommandé" in reply
    assert "8986" in reply or "8 986" in reply
