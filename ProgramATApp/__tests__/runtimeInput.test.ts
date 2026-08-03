import {
  buildRuntimeInputsPayload,
  normalizeRuntimeInputValue,
} from '../runtimeInput';

describe('runtime input helpers', () => {
  test('normalizes whitespace before commit', () => {
    expect(normalizeRuntimeInputValue('  water   cup  ')).toBe('water cup');
  });

  test('builds an empty payload when no committed value exists', () => {
    expect(
      buildRuntimeInputsPayload(
        {
          key: 'target_object',
          label: 'Object to find',
          placeholder: 'Enter an object',
          prompt_instruction: 'Focus on {value}.',
        },
        '',
      ),
    ).toEqual({});
  });

  test('builds a keyed payload for committed runtime input', () => {
    expect(
      buildRuntimeInputsPayload(
        {
          key: 'target_object',
          label: 'Object to find',
          placeholder: 'Enter an object',
          prompt_instruction: 'Focus on {value}.',
        },
        ' water cup ',
      ),
    ).toEqual({target_object: 'water cup'});
  });
});
