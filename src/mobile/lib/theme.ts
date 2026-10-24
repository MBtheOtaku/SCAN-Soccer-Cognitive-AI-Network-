import { StyleSheet } from 'react-native';

export const colors = {
  background: '#070C09', panel: '#101B14', raised: '#17251B', line: '#283D2E',
  green: '#91F5A2', text: '#F2F7EF', muted: '#A2B1A4', dark: '#0B1C10', error: '#FFB6A8',
};

export const ui = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background },
  content: { padding: 24, paddingBottom: 36, gap: 22 },
  eyebrow: { color: colors.green, fontSize: 11, letterSpacing: 2, fontWeight: '800' },
  title: { color: colors.text, fontSize: 34, lineHeight: 39, fontWeight: '800', letterSpacing: -1 },
  body: { color: colors.muted, fontSize: 15, lineHeight: 23 },
  small: { color: colors.muted, fontSize: 12, lineHeight: 18 },
  card: { backgroundColor: colors.panel, borderWidth: 1, borderColor: colors.line, borderRadius: 22, padding: 20, gap: 12 },
  row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 12 },
  heading: { color: colors.text, fontSize: 19, fontWeight: '700' },
  badge: { color: colors.green, fontSize: 10, letterSpacing: 1.5, fontWeight: '800' },
  error: { color: colors.error, fontSize: 14, lineHeight: 21 },
});
