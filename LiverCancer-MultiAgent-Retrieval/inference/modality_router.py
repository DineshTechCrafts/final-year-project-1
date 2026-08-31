class ModalityRouter:
    """
    Strictly gates requests based on modality, blocking MRI from unvalidated inference.
    """
    def route(self, modality: str) -> str:
        modality = modality.upper()
        if modality == "MRI":
            return "MRI_PIPELINE_NOT_VALIDATED"
        if modality == "CT":
            return "CT_PIPELINE_OK"
        return "UNSUPPORTED_MODALITY"
