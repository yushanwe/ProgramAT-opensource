export interface RuntimeInputDefinition {
  key: string;
  label: string;
  placeholder: string;
  prompt_instruction: string;
}

export function normalizeRuntimeInputValue(value: string): string {
  return value.replace(/\s+/g, ' ').trim();
}

export function buildRuntimeInputsPayload(
  definition?: RuntimeInputDefinition,
  committedValue?: string,
): Record<string, string> {
  if (!definition) {
    return {};
  }
  const normalized = normalizeRuntimeInputValue(committedValue || '');
  if (!normalized) {
    return {};
  }
  return {
    [definition.key]: normalized,
  };
}
