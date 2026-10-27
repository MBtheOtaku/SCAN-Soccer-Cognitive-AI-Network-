import { useRef, useState } from 'react';
import { Text, View } from 'react-native';
import * as Sharing from 'expo-sharing';
import { Action } from './controls';
import type { Rep } from '../lib/reps';
import { ui } from '../lib/theme';

export function RepExport({ rep }: { rep: Rep }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const locked = useRef(false);
  async function share() {
    if (locked.current) return;
    locked.current = true; setBusy(true); setError(null);
    try {
      if (!await Sharing.isAvailableAsync()) throw new Error('Video sharing is unavailable on this device.');
      const mov = rep.name.endsWith('.mov');
      await Sharing.shareAsync(rep.uri, { dialogTitle: 'Export SCAN rep',
        mimeType: mov ? 'video/quicktime' : 'video/mp4',
        UTI: mov ? 'com.apple.quicktime-movie' : 'public.mpeg-4' });
      // Dismissing the share sheet is not proof that the user exported a copy.
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not open the share menu. Try again.'); }
    finally { locked.current = false; setBusy(false); }
  }
  return <View style={{ gap: 10 }}>
    <Action title={busy ? 'Opening share menu…' : 'Export video'} secondary disabled={busy} onPress={() => { void share(); }} />
    <Text style={ui.small}>Choose Save Video or Save to Files in the share menu. Your original stays in SCAN. Titles and notes stay in the app.</Text>
    {error && <Text accessibilityLiveRegion="polite" style={ui.error}>{error}</Text>}
  </View>;
}
