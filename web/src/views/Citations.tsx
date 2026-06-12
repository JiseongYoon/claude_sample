// Citations — renders any sources extracted from an answer's text (). Display-only, escaped
// (React children — never a live <a> link, since the answer is untrusted). Nothing rendered when absent.
import { extractSources } from "../lib/citations";
import styles from "./Chat.module.css";

export function Citations({ text }: { text: string }) {
 const sources = extractSources(text);
 if (sources.length === 0) return null;
 return (
 <div className={styles.citations} data-testid="citations">
 <small>Sources</small>
 <ul className={styles.citationsList}>
 {sources.map((s, i) => (
 <li key={i} className={styles.citation} data-testid="citation">
 {s}
 </li>
 ))}
 </ul>
 </div>
 );
}
