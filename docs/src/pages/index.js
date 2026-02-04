import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  return (
    <header className={styles.heroBanner}>
      <div className="container">
        <h1 className={styles.heroTitle}>
          Dendrite
        </h1>
        <p className={styles.heroSubtitle}>Open-source brain-computer interface application</p>
        <div className={styles.buttons}>
          <Link className={`button button--lg ${styles.accentButton}`} to="/docs/quickstart">
            Get Started
          </Link>
        </div>
      </div>
    </header>
  );
}

function Features() {
  const features = [
    {
      title: 'Data Acquisition',
      description: 'Stream from any LSL-compatible device with synchronized multimodal support.'
    },
    {
      title: 'Machine Learning',
      description: 'Train neural networks or classical decoders, deploy for real-time inference.'
    },
    {
      title: 'User Application',
      description: 'Stream predictions to external apps via LSL, ROS2, or sockets.'
    }
  ];

  return (
    <section className={styles.featuresSection}>
      <div className="container">
        <div className={styles.featuresGrid}>
          {features.map((feature, idx) => (
            <div key={idx} className={styles.featureCard}>
              <h3 className={styles.featureTitle}>{feature.title}</h3>
              <p className={styles.featureDescription}>{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <Layout
      title="Dendrite"
      description="Real-time neural signal processing and brain-computer interfaces"
      wrapperClassName="homepage-wrapper">
      <HomepageHeader />
      <main>
        <Features />
      </main>
    </Layout>
  );
}