// Menu boczne i motyw bez Reacta (strona statyczna na GitHub Pages).
(function () {
  var aside = document.querySelector('aside')
  var guzik = document.querySelector('button[aria-label="Open menu"]')
  if (aside && guzik) {
    guzik.addEventListener('click', function () {
      var otwarte = aside.classList.toggle('translate-x-0')
      aside.classList.toggle('-translate-x-full', !otwarte)
      guzik.setAttribute('aria-label', otwarte ? 'Close menu' : 'Open menu')
    })
  }
  var motyw = document.querySelector('button[aria-label="Choose theme"]')
  if (!motyw) return
  var kolejka = ['auto', 'light', 'dark']
  var nazwy = { auto: 'Auto', light: 'Light', dark: 'Dark' }
  var napis = motyw.querySelector('[data-slot="select-value"]') || motyw.querySelector('span')
  var teraz = 'auto'
  try { teraz = localStorage.getItem('payload-theme') || 'auto' } catch (e) {}
  if (napis) napis.textContent = nazwy[teraz]
  motyw.addEventListener('click', function () {
    teraz = kolejka[(kolejka.indexOf(teraz) + 1) % 3]
    var efekt = teraz === 'auto' ? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') : teraz
    try { teraz === 'auto' ? localStorage.removeItem('payload-theme') : localStorage.setItem('payload-theme', teraz) } catch (e) {}
    document.documentElement.setAttribute('data-theme', efekt)
    if (napis) napis.textContent = nazwy[teraz]
  })
})()
