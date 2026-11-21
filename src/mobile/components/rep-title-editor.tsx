import { useRef, useState } from 'react';
import { Keyboard, StyleSheet, Text, TextInput, View } from 'react-native';
import { Action } from './controls';
import { useReps } from '../lib/rep-store';
import { REP_NOTES_LIMIT, REP_TITLE_LIMIT, type Rep } from '../lib/reps';
import { colors, ui } from '../lib/theme';
import { TrainingIntentionFields } from './training-intention';
import { intentionLabel, normalizeIntention, outcomeLabels, type TrainingIntention } from '../lib/training-intention';

export function RepTitleEditor({ rep }: { rep: Rep }) {
  const { updateDetails } = useReps();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const [notes, setNotes] = useState('');
  const [intention, setIntention] = useState<TrainingIntention>(() => normalizeIntention());
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const busy = useRef(false);

  async function save() {
    if (busy.current) return;
    busy.current = true; setSaving(true); setError(null);
    try {
      await updateDetails(rep.name, draft, notes, intention);
      Keyboard.dismiss(); setEditing(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not save rep details. Try again.');
    } finally { busy.current = false; setSaving(false); }
  }

  if (!editing) return <View style={{ gap: 12 }}>
    <View style={ui.card}><Text style={ui.badge}>TRAINING INTENTION</Text><Text style={ui.heading}>{intentionLabel(rep.trainingIntention)}</Text>
      {rep.trainingIntention?.reported_outcome && <Text style={ui.small}>Player-reported: {outcomeLabels[rep.trainingIntention.reported_outcome]} · Not verified by SCAN</Text>}
    </View>
    {rep.notes && <View style={ui.card}><Text style={ui.badge}>YOUR NOTES</Text><Text style={ui.body}>{rep.notes}</Text></View>}
    <Action title="Edit rep details & intention" secondary onPress={() => {
      setDraft(rep.title || ''); setNotes(rep.notes || ''); setIntention(normalizeIntention(rep.trainingIntention)); setError(null); setEditing(true);
    }} />
  </View>;

  return <View style={ui.card}>
    <Text style={ui.heading}>Rep details</Text>
    <TrainingIntentionFields value={intention} onChange={setIntention} disabled={saving} includeOutcome />
    <Text style={ui.small}>Title</Text>
    <TextInput accessibilityLabel="Rep title" value={draft} onChangeText={setDraft}
      placeholder="e.g. Left-foot shot" placeholderTextColor={colors.muted}
      maxLength={REP_TITLE_LIMIT} editable={!saving} autoFocus returnKeyType="done"
      onSubmitEditing={() => { void save(); }} style={s.input} selectionColor={colors.green} />
    <Text style={ui.small}>{draft.length}/{REP_TITLE_LIMIT} · Leave blank to use the default label.</Text>
    <Text style={ui.small}>Your notes</Text>
    <TextInput accessibilityLabel="Rep notes" value={notes} onChangeText={setNotes}
      placeholder="What were you practising? How did it feel?" placeholderTextColor={colors.muted}
      maxLength={REP_NOTES_LIMIT} editable={!saving} multiline textAlignVertical="top"
      style={[s.input, { minHeight: 120 }]} selectionColor={colors.green} />
    <Text style={ui.small}>{notes.length}/{REP_NOTES_LIMIT} · These are your observations, not SCAN analysis.</Text>
    {error && <Text accessibilityLiveRegion="polite" style={ui.error}>{error}</Text>}
    <Action title={saving ? 'Saving details…' : 'Save details'} disabled={saving} onPress={() => { void save(); }} />
    <Action title="Cancel" secondary disabled={saving} onPress={() => { Keyboard.dismiss(); setEditing(false); }} />
  </View>;
}

const s = StyleSheet.create({
  input: { color: colors.text, backgroundColor: colors.background, borderColor: colors.line,
    borderWidth: 1, borderRadius: 12, padding: 14, fontSize: 17, minHeight: 52 },
});
