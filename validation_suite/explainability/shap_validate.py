import numpy as np
import shap

def shap_top_features(pipeline_model, X_sample, topn=25) -> dict:
    pre = pipeline_model.named_steps.get("pre", None)
    clf = pipeline_model.named_steps.get("clf", None)
    if pre is None or clf is None:
        return {"error": "model is not Pipeline(pre+clf)"}

    X_trans = pre.transform(X_sample)

    try:
        feature_names = pre.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_trans.shape[1])]

    try:
        explainer = shap.Explainer(clf, X_trans)
        try:
            sv = explainer(X_trans, check_additivity=False)
        except TypeError:
            sv = explainer(X_trans)

        vals = np.abs(sv.values)
        if vals.ndim == 3:
            vals = vals[:, :, -1]
        imp = vals.mean(axis=0)

        idx = np.argsort(-imp)[:topn]
        top = [{"feature": feature_names[i], "mean_abs_shap": float(imp[i])} for i in idx]
        return {"top_features": top}
    except Exception as e:
        return {"error": f"shap failed: {repr(e)}"}
