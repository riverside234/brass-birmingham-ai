"""Lossless, versioned storage for explicit self-play source approvals."""
import copy
import json

from brass_ai.learning.contracts import require


EXPANDED_FORMAT = 'selfplay-resume-sources/1.0.0'
COMPACT_FORMAT = 'selfplay-resume-sources/2.0.0'
SHARED_FIELDS = ('reason', 'config_changes', 'config_migrations',
                 'config_migration', 'learning_compatibility')
FIELD_TYPES = (str, dict, list, str, str)


def expand_source_reviews(receipt):
    """Resolve explicit references without changing approval order or contents.

    Both versions return independent, ordinary v1 records. References are
    reserved fields: a missing or ambiguous reference must never silently drop
    a configuration restriction or select a different migration.
    """
    require(isinstance(receipt, dict) and receipt.get('format') in
            (EXPANDED_FORMAT, COMPACT_FORMAT), 'Unsupported source review receipt')
    require(isinstance(receipt.get('transitions'), list), 'Invalid source review transitions')
    compact = receipt['format'] == COMPACT_FORMAT
    definitions = receipt.get('definitions', {})
    if compact:
        require(isinstance(definitions, dict) and set(definitions) <= set(SHARED_FIELDS),
                'Invalid source review definitions')
        for field, kind in zip(SHARED_FIELDS, FIELD_TYPES):
            table = definitions.get(field, {})
            require(isinstance(table, dict) and all(isinstance(key, str) and isinstance(value, kind)
                    for key, value in table.items()), f'Invalid source review definition: {field}')
    else:
        require('definitions' not in receipt, 'Unexpected source review definitions')
    transitions = []
    for entry in receipt['transitions']:
        require(isinstance(entry, dict), 'Invalid source review transition')
        expanded = {}
        for key, value in entry.items():
            if key.endswith('_ref'):
                field = key[:-4]
                require(compact and field in SHARED_FIELDS and field not in entry,
                        f'Invalid or ambiguous source review reference: {key}')
                table = definitions.get(field, {})
                require(isinstance(value, str) and value in table,
                        f'Missing source review definition: {key}')
                expanded[field] = copy.deepcopy(table[value])
            else:
                require(not compact or key not in SHARED_FIELDS,
                        f'Compact source review requires a reference: {key}')
                expanded[key] = copy.deepcopy(value)
        require(all(isinstance(expanded.get(field), str) for field in
                    ('from_implementation', 'to_implementation', 'learning_compatibility')),
                'Missing source review identity')
        transitions.append(expanded)
    result = copy.deepcopy({key: value for key, value in receipt.items()
                            if key not in ('definitions', 'transitions')})
    result.update(format=EXPANDED_FORMAT, transitions=transitions)
    return result


def compact_source_reviews(receipt):
    """Intern repeated values, retaining every approval and all metadata.

    First-use numbering makes output deterministic and leaves existing IDs
    stable when approvals are appended. No implicit/transitive approvals are
    introduced by this storage format.
    """
    expanded = expand_source_reviews(receipt)
    definitions = {field: {} for field in SHARED_FIELDS}
    identifiers = {field: {} for field in SHARED_FIELDS}
    transitions = []
    for entry in expanded['transitions']:
        compact = {}
        for field, value in entry.items():
            if field not in definitions:
                compact[field] = value
                continue
            # Preserve nested key order and numeric representations as well as
            # values; sorting or Python equality could merge 1 with 1.0.
            key = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            table = identifiers[field]
            if key not in table:
                table[key] = str(len(table))
                definitions[field][table[key]] = value
            compact[field + '_ref'] = table[key]
        transitions.append(compact)
    result = dict(expanded, format=COMPACT_FORMAT, definitions=definitions, transitions=transitions)
    # Validate the output too, including any malformed inline legacy values.
    require(expand_source_reviews(result) == expanded, 'Source review compaction changed approvals')
    return result
