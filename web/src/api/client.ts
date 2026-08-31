export const API_BASE_URL = "http://localhost:8000/api";

export const fetchAPI = async (endpoint: string) => {
  const response = await fetch(`${API_BASE_URL}${endpoint}`);
  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }
  return response.json();
};

export const getImageUrl = (patientId: string, sliceIndex: number) => {
  return `${API_BASE_URL}/patients/${patientId}/slice/${sliceIndex}/image`;
};

export const getMaskUrl = (patientId: string, sliceIndex: number) => {
  return `${API_BASE_URL}/patients/${patientId}/slice/${sliceIndex}/mask`;
};

export const getOverlayUrl = (patientId: string, sliceIndex: number, activeClasses: number[]) => {
  const classParam = activeClasses.join(",");
  return `${API_BASE_URL}/patients/${patientId}/slice/${sliceIndex}/overlay?classes=${classParam}`;
};

export const uploadImage = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);
  
  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }
  return response.json();
};
