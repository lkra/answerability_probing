export const meta = {
  name: 'translate-silver-references',
  description: 'Translate 80 English silver reference answers into 23 languages, grounded in authoritative passages where available',
  phases: [
    { title: 'Translate', detail: 'one agent per language group, writes <lang>.json' },
  ],
}

// Paths relative to the repository root, where the workflow is run from.
const IN = 'data/silver/translation_inputs'
const OUT = 'data/silver/translation_outputs'

const GROUPS = [
  ['arb', 'ben'], ['bho', 'ces'], ['ckb', 'deu'], ['fin', 'hat'],
  ['hin', 'ind'], ['ita', 'jpn'], ['kor', 'mar'], ['pes', 'por'],
  ['rus', 'slk'], ['spa', 'tur'], ['vie', 'yor'], ['zho'],
]

const NAMES = {
  arb: 'Modern Standard Arabic', ben: 'Bengali', bho: 'Bhojpuri', ces: 'Czech',
  ckb: 'Kurdish (see note)', deu: 'German', fin: 'Finnish', hat: 'Haitian Creole',
  hin: 'Hindi', ind: 'Indonesian', ita: 'Italian', jpn: 'Japanese', kor: 'Korean',
  mar: 'Marathi', pes: 'Persian (Farsi)', por: 'Portuguese', rus: 'Russian',
  slk: 'Slovak', spa: 'Spanish', tur: 'Turkish', vie: 'Vietnamese',
  yor: 'Yoruba', zho: 'Chinese (Simplified)',
}

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['lang', 'n_translated', 'n_grounded', 'uncertain'],
  properties: {
    lang: { type: 'string' },
    n_translated: { type: 'integer' },
    n_grounded: { type: 'integer', description: 'units where wording was lifted verbatim from target_passage' },
    uncertain: {
      type: 'array',
      description: 'item_ids you are genuinely unsure about, with why (empty if none)',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['item_id', 'why'],
        properties: { item_id: { type: 'string' }, why: { type: 'string' } },
      },
    },
    notes: { type: 'string', description: 'anything the caller should know about this language' },
  },
}

function prompt(lang) {
  return `You are producing REFERENCE ANSWERS for a multilingual QA benchmark. These are
scoring references, so faithfulness matters far more than fluency or elegance.

Read ${IN}/${lang}.json. It has a "units" array; each unit has:
  - english_answer   : the reference answer, in English, that you must render into ${NAMES[lang]}
  - english_question : the question it answers (context only -- do NOT translate this)
  - english_passage  : the source passage the answer was taken from
  - target_passage   : the SAME source text in ${NAMES[lang]}, or null
  - same_language    : true when a target_passage is present

Write ${OUT}/${lang}.json with the identical structure and unit order, filling in each
unit's "translation" field. Change nothing else. Create the directory if needed.

RULES, in priority order:

1. If target_passage is NOT null, it is an OFFICIAL translation of english_passage
   (UDHR, the UNESCO 2003 Convention, EU treaty/charter/AI-Act text, or a Wikipedia
   article). Locate the span in target_passage that corresponds to english_answer and
   REUSE ITS EXACT WORDING, verbatim, including that text's terminology and
   inflection. Do not retranslate it and do not "improve" it. This is the single most
   important rule: for these units the benchmark expects a passage span, and a fluent
   paraphrase would be scored as wrong. Adjust only what grammar strictly requires to
   make it a standalone sentence.

2. If target_passage IS null, translate english_answer into ${NAMES[lang]} faithfully,
   staying close to english_passage's phrasing and register rather than paraphrasing.
   Prefer the standard/official ${NAMES[lang]} terminology for these well-known
   instruments where you know it.

3. Keep the answer to ONE sentence, matching the length and shape of english_answer.
   Several answers are short noun phrases ("Simplicity.", "Terrorist groups.") -- keep
   those short; do not expand them into full sentences.

4. Output ONLY the answer text. No explanation, no transliteration, no English gloss,
   no quotation marks around it.

5. Use the script and orthography that ${NAMES[lang]} actually uses in target_passage
   where one is present.${lang === 'ckb' ? `

   IMPORTANT for this language: the test set labels this "ckb" (Sorani), but its
   passages are actually KURMANJI Kurdish in Latin script, while the questions are
   Sorani in Arabic script. Match the passages: produce LATIN-SCRIPT KURMANJI, and say
   so in your notes.` : ''}

Be honest in the "uncertain" field. If you do not have solid command of ${NAMES[lang]},
or a specific term is a guess, list those item_ids rather than quietly emitting something
plausible-looking -- a silently wrong reference corrupts the benchmark score for an entire
language, which is worse than a flagged gap.

Verify before finishing: the file parses as JSON, has the same number of units in the
same order, and every "translation" is non-null and in ${NAMES[lang]}.`
}

phase('Translate')

const results = await parallel(
  GROUPS.map(group => async () => {
    const out = []
    for (const lang of group) {
      const r = await agent(prompt(lang), {
        label: `translate:${lang}`,
        phase: 'Translate',
        schema: SCHEMA,
      })
      if (r) out.push(r)
    }
    return out
  })
)

const flat = results.filter(Boolean).flat()
log(`${flat.length}/23 languages returned`)

return {
  languages: flat.length,
  total_translated: flat.reduce((a, r) => a + (r.n_translated || 0), 0),
  total_grounded: flat.reduce((a, r) => a + (r.n_grounded || 0), 0),
  per_language: flat.map(r => ({
    lang: r.lang,
    n: r.n_translated,
    grounded: r.n_grounded,
    uncertain: (r.uncertain || []).length,
    notes: r.notes,
  })),
  uncertain: flat.flatMap(r => (r.uncertain || []).map(u => ({ lang: r.lang, ...u }))),
}
