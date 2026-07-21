import openpyxl

wb = openpyxl.load_workbook(r'docs/GCER Population Tables.xlsx', data_only=True)
ws = wb['Current Processing']
rows = list(ws.iter_rows(values_only=True))
hdr = [('' if v is None else str(v).strip()) for v in rows[0]]


def find(*keys):
    for i, h in enumerate(hdr):
        if all(k.lower() in (h or '').lower() for k in keys):
            return i
    return None


i_ds = find('Dataset', 'Location') or 0
i_cond = find('Construct')
i_local = find('Local', 'Movie', 'Directory')
i_cs = find('cS', 'Directory')
i_mov = find('Movie', 'Directory')  # last one
# find the *last* 'Movie Directory' (there are two)
mov_idxs = [i for i, h in enumerate(hdr) if 'movie directory' in (h or '').lower()]
i_movlast = mov_idxs[-1] if mov_idxs else i_mov
i_emp = find('EMPIAR')

for r in rows[1:]:
    vals = [('' if v is None else str(v).strip()) for v in r]
    ds = vals[i_ds] if i_ds < len(vals) else ''
    cond = vals[i_cond] if i_cond is not None and i_cond < len(vals) else ''
    if not (ds or cond):
        continue

    def g(i):
        return vals[i] if (i is not None and i < len(vals)) else ''
    print(f'{ds[:20]:<20} | {cond[:42]:<42}')
    for name, i in (('local', i_local), ('cS', i_cs), ('movdir', i_movlast),
                    ('EMPIAR', i_emp)):
        v = g(i)
        if v:
            print(f'      {name}: {v[:110]}')
