import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import styles from "./styles/socialbar.scss"

export default (() => {
  function SocialBar(props: QuartzComponentProps) {
    return (
      <div class="socialbar">
        <a href="mailto:nguyenxuankhanhm@gmail.com" target="_blank" rel="noopener noreferrer">📧</a>
        <a href="https://www.linkedin.com/in/khanh-nguyen-3a8913b3" target="_blank" rel="noopener noreferrer">💼</a>
        <a href="https://x.com/khanhxuannguyen" target="_blank" rel="noopener noreferrer">𝕏</a>
        <a href="https://scholar.google.com/citations?user=SmqouhIAAAAJ&hl=en" target="_blank" rel="noopener noreferrer">🎓</a>
        <a href="https://github.com/YOUR-HANDLE" target="_blank" rel="noopener noreferrer">🐙</a>
      </div>
    )
  }

  SocialBar.css = styles
  return SocialBar
}) satisfies QuartzComponentConstructor


