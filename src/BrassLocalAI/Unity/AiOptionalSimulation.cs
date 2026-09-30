using System;
using System.Collections.Generic;
using UnityEngine;

namespace BrassLegacyAI
{
    // Opt-in marker for decorative work. Gameplay/replay/UI objects are never
    // auto-selected. Designers can register custom physics controllers as Scripts.
    public sealed class AiOptionalSimulation : MonoBehaviour, IAiOptionalWork
    {
        public ParticleSystem[] Particles = new ParticleSystem[0];
        public Behaviour[] Scripts = new Behaviour[0];
        IDisposable registration;
        void OnEnable()
        {
            var token = AiExecutionResources.Register(this);
            // Registration can immediately suspend work and deactivate its owner.
            if (this == null || !isActiveAndEnabled) token.Dispose(); else registration = token;
        }
        void OnDisable() { registration?.Dispose(); registration = null; }
        void OnDestroy() { registration?.Dispose(); registration = null; }
        public IDisposable Suspend()
        {
            var snapshot = new Snapshot();
            try
            {
                foreach (var particle in Particles ?? new ParticleSystem[0])
                    if (particle != null && particle.isPlaying) { snapshot.Particles.Add(particle); particle.Pause(false); }
                foreach (var script in Scripts ?? new Behaviour[0])
                    if (script != null && !(script is AiOptionalSimulation) && !(script is AiResourceDriver) && script.enabled)
                    { snapshot.Scripts.Add(script); script.enabled = false; }
                return snapshot;
            }
            catch { snapshot.Dispose(); throw; }
        }
        sealed class Snapshot : IDisposable
        {
            internal readonly List<ParticleSystem> Particles = new List<ParticleSystem>();
            internal readonly List<Behaviour> Scripts = new List<Behaviour>();
            public void Dispose()
            {
                foreach (var particle in Particles) if (particle != null) particle.Play(false);
                foreach (var script in Scripts) if (script != null) script.enabled = true;
                Particles.Clear(); Scripts.Clear();
            }
        }
    }
}
