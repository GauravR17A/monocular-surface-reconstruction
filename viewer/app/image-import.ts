export const IMAGE_ACCEPT = '.jpg,.jpeg,.png,.tif,.tiff,.webp,.bmp,.jp2,image/jpeg,image/png,image/tiff,image/webp,image/bmp,image/jp2';
export const IMAGE_FORMATS = 'GeoTIFF / TIFF, PNG, JPG, WebP, BMP or JP2';
export const MAX_IMAGE_BYTES = 512 * 1024 * 1024;
const extensions = new Set(['jpg', 'jpeg', 'png', 'tif', 'tiff', 'webp', 'bmp', 'jp2']);
const mimeExtensions: Record<string, string> = {
  'image/jpeg': 'jpg', 'image/png': 'png', 'image/tiff': 'tif', 'image/x-tiff': 'tif',
  'image/webp': 'webp', 'image/bmp': 'bmp', 'image/x-ms-bmp': 'bmp', 'image/jp2': 'jp2',
};

/** Keep original GeoTIFF bytes and names; MIME values from Explorer are often empty. */
export function prepareImageImport(file: File): File {
  if (file.size === 0) throw new Error(`${file.name || 'Image'} is empty.`);
  if (file.size > MAX_IMAGE_BYTES) throw new Error(`${file.name} exceeds 512 MB. Crop or resample it first.`);
  const extension = file.name.match(/\.([^.]+)$/)?.[1].toLowerCase();
  if (extension && extensions.has(extension)) return file;
  const inferred = mimeExtensions[file.type.toLowerCase()];
  if (!extension && inferred) return new File([file], `${file.name || 'clipboard-image'}.${inferred}`, { type: file.type, lastModified: file.lastModified });
  throw new Error(`${file.name || 'This file'} is unsupported. Choose ${IMAGE_FORMATS}.`);
}

export function filesFromTransfer(transfer: DataTransfer | null): File[] {
  if (!transfer) return [];
  const files = Array.from(transfer.files);
  return files.length ? files : Array.from(transfer.items).flatMap(item => {
    const file = item.kind === 'file' ? item.getAsFile() : null;
    return file ? [file] : [];
  });
}

export function clipboardImageType(types: readonly string[]) {
  return ['image/tiff', 'image/png', 'image/jpeg', 'image/webp', 'image/bmp', 'image/jp2'].find(type => types.includes(type));
}

export function clipboardImageFile(blob: Blob, index = 0) {
  return new File([blob], `clipboard-${Date.now()}-${index + 1}.${mimeExtensions[blob.type] ?? 'png'}`, { type: blob.type });
}
